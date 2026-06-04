"""AnimeCam — Shinkai-style neural webcam for Twitch streaming.

Full-frame pipeline: scale down → Shinkai model → scale back up.

Controls:  M mirror  |  ESC quit
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CAMERA_ID   = 0
MODEL_PATH  = "models/AnimeGANv3_Shinkai.onnx"
INFER_LONG  = 256    # long-edge inference resolution; preserves aspect ratio
BLEND_ALPHA = 0.95   # how much of the stylised result shows (0=original, 1=full style)

_OPENVINO = "OpenVINOExecutionProvider" in ort.get_available_providers()


# ── Session ───────────────────────────────────────────────────────────────────

def make_session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if _OPENVINO:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: list = [("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}),
                           "CPUExecutionProvider"]
    else:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# ── Inference size (aspect-ratio preserving) ──────────────────────────────────

def infer_size(fh: int, fw: int) -> tuple[int, int]:
    """Scale so the long edge == INFER_LONG, keep aspect ratio."""
    scale = INFER_LONG / max(fh, fw)
    return int(fh * scale), int(fw * scale)


# ── Pre / post — AnimeGANv3: NHWC, range [−1, 1] ─────────────────────────────

def preprocess(frame_bgr: np.ndarray, ih: int, iw: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    t   = rgb.astype(np.float32) / 127.5 - 1.0
    return t[np.newaxis]                              # [1, ih, iw, 3]


def postprocess(raw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    t   = (raw[0] + 1.0) * 127.5                     # [ih, iw, 3]
    t   = np.clip(t, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


# ── Inference thread ──────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    def __init__(self, sess: ort.InferenceSession) -> None:
        super().__init__(daemon=True)
        self._sess     = sess
        self._inp_name = sess.get_inputs()[0].name

        self._in_lock  = threading.Lock()
        self._in_frame: np.ndarray | None = None
        self._in_seq   = 0
        self._proc_seq = -1

        self._out_lock  = threading.Lock()
        self._out_frame: np.ndarray | None = None

        self.inf_fps  = 0.0
        self._running = threading.Event()
        self._running.set()
        self._wakeup  = threading.Event()

    def push(self, frame: np.ndarray) -> None:
        with self._in_lock:
            self._in_frame = frame
            self._in_seq  += 1
        self._wakeup.set()

    def get_result(self) -> np.ndarray | None:
        with self._out_lock:
            return self._out_frame

    def stop(self) -> None:
        self._running.clear()
        self._wakeup.set()

    def run(self) -> None:
        t_prev = time.perf_counter()
        while self._running.is_set():
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()

            with self._in_lock:
                if self._in_seq == self._proc_seq or self._in_frame is None:
                    continue
                frame          = self._in_frame
                self._proc_seq = self._in_seq

            fh, fw  = frame.shape[:2]
            ih, iw  = infer_size(fh, fw)

            inp    = preprocess(frame, ih, iw)
            raw    = self._sess.run(None, {self._inp_name: inp})[0]
            styled = postprocess(raw, fh, fw)

            result = cv2.addWeighted(styled, BLEND_ALPHA, frame, 1 - BLEND_ALPHA, 0)

            with self._out_lock:
                self._out_frame = result

            t_now      = time.perf_counter()
            self.inf_fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev     = t_now


# ── OSD ───────────────────────────────────────────────────────────────────────

def draw_osd(img: np.ndarray, d_fps: float, i_fps: float,
             mirrored: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    ov = out.copy()
    cv2.rectangle(ov, (0, h-46), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.4, out, 0.6, 0, out)

    label = "Shinkai" + ("  [mirror]" if mirrored else "")
    cv2.putText(out, label, (14, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 210, 210), 2, cv2.LINE_AA)

    backend = "OpenVINO" if _OPENVINO else "CPU"
    cv2.putText(out, f"disp {d_fps:.0f}  infer {i_fps:.0f} fps  [{backend}]",
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 80), 2, cv2.LINE_AA)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    print(f"Loading {MODEL_PATH}  [{('OpenVINO' if _OPENVINO else 'CPU')}]…")
    sess = make_session(MODEL_PATH)
    print("Ready.  Controls: M mirror | ESC quit\n")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    worker = InferenceThread(sess)
    worker.start()

    mirrored = False
    t_prev   = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirrored:
            frame = cv2.flip(frame, 1)

        worker.push(frame)

        result  = worker.get_result()
        display = result if result is not None else frame

        t_now  = time.perf_counter()
        d_fps  = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.imshow("AnimeCam", draw_osd(display, d_fps, worker.inf_fps, mirrored))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('m'), ord('M')):
            mirrored = not mirrored

    worker.stop()
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
