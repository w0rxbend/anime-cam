"""
AnimeCam
========
Real-time Shinkai-style neural webcam effect for Twitch streaming.

Architecture
------------
Two threads share work so the display never blocks on inference:

    Camera (main thread)
        │  push(frame)
        ▼
    InferenceThread          ← always processes the *newest* frame,
        │  get_result()         drops any queued frames it can't keep up with
        ▼
    Display (main thread)   ← shows last stylised frame at full camera FPS

Model
-----
AnimeGANv3 Shinkai (NHWC, float32, range [−1, 1])
  • Originally a TensorFlow/Keras model, exported to ONNX.
  • Input  : [1, H, W, 3]  RGB  values normalised to [−1, 1]
  • Output : [1, H, W, 3]  RGB  values in [−1, 1]  (same spatial dims)
  • Dynamic spatial dimensions → can accept any (H, W).
  • At 256×144 (1280×720 scaled down) inference is ~14 fps on CPU with OpenVINO.

Inference resolution
--------------------
The full 1280×720 frame is scaled so its *long edge* equals INFER_LONG,
preserving aspect ratio (→ 256×144).  The stylised result is then upscaled
back to 1280×720 with bicubic interpolation before display.

Performance knobs
-----------------
INFER_LONG   lower  → faster inference, softer output
             higher → slower inference, sharper style detail
BLEND_ALPHA  1.0    → full Shinkai style
             0.0    → original camera feed (passthrough)

Controls
--------
  M / m   toggle horizontal mirror
  ESC     quit

TODO: Virtual camera output
---------------------------
Stream the stylised frames to a v4l2loopback virtual webcam device so that
OBS, Discord, Zoom, and other apps can pick up the AnimeCam feed as a regular
camera source.

Implementation outline:

  1. Load the kernel module (once, outside the app):
       sudo modprobe v4l2loopback devices=1 video_nr=10 \\
           card_label="AnimeCam" exclusive_caps=1

  2. Open the device for writing and configure the pixel format via ioctl:
       import fcntl, ctypes
       VIDIOC_S_FMT = 0xC0D05605        # from <linux/videodev2.h>

       class v4l2_pix_format(ctypes.Structure):
           _fields_ = [("width", ctypes.c_uint32),
                       ("height", ctypes.c_uint32),
                       ("pixelformat", ctypes.c_uint32),   # V4L2_PIX_FMT_BGR24 = 0x33524742
                       ("field", ctypes.c_uint32),
                       ("bytesperline", ctypes.c_uint32),
                       ("sizeimage", ctypes.c_uint32),
                       ("colorspace", ctypes.c_uint32),
                       ("priv", ctypes.c_uint32)]

       class v4l2_format(ctypes.Structure):
           _fields_ = [("type", ctypes.c_uint32),          # V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
                       ("fmt", v4l2_pix_format)]

       fd = open("/dev/video10", "wb", buffering=0)
       fmt = v4l2_format()
       fmt.type                 = 2
       fmt.fmt.width            = 1280
       fmt.fmt.height           = 720
       fmt.fmt.pixelformat      = 0x33524742  # BGR24
       fmt.fmt.field            = 1           # V4L2_FIELD_NONE
       fmt.fmt.bytesperline     = 1280 * 3
       fmt.fmt.sizeimage        = 1280 * 720 * 3
       fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)

  3. In the output loop, write each BGR frame as raw bytes:
       def write_vcam(fd, frame_bgr: np.ndarray) -> None:
           if not frame_bgr.flags['C_CONTIGUOUS']:
               frame_bgr = np.ascontiguousarray(frame_bgr)
           fd.write(frame_bgr.tobytes())

  4. Dependency: v4l2loopback-dkms kernel module.
       Arch:   yay -S v4l2loopback-dkms
       Ubuntu: sudo apt install v4l2loopback-dkms

  5. Integration point: call write_vcam() inside InferenceThread.run()
     just after updating self._out_frame, or add a dedicated third thread
     that reads get_result() and writes to the device.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ── Configuration ─────────────────────────────────────────────────────────────

CAMERA_ID   = 0
MODEL_PATH  = "models/AnimeGANv3_Shinkai.onnx"

# Long-edge resolution used for inference.
# 1280×720 → 256×144 at INFER_LONG=256.  Raise to 384 or 512 for sharper
# style at the cost of lower inference FPS.
INFER_LONG  = 256

# Blend factor between the stylised and original frame (applied full-frame).
# 1.0 = pure Shinkai style.  Reduce slightly if the effect feels too strong.
BLEND_ALPHA = 0.95

_OPENVINO = "OpenVINOExecutionProvider" in ort.get_available_providers()


# ── ONNX session ──────────────────────────────────────────────────────────────

def make_session(path: str) -> ort.InferenceSession:
    """Create an ORT session, preferring OpenVINO CPU EP for ~30 % speedup."""
    opts = ort.SessionOptions()
    if _OPENVINO:
        # OpenVINO performs its own graph optimisation pass; disable ORT's to
        # avoid conflicts between the two optimisation pipelines.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: list = [
            ("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}),
            "CPUExecutionProvider",   # fallback for any unsupported ops
        ]
    else:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# ── Spatial helpers ───────────────────────────────────────────────────────────

def infer_size(fh: int, fw: int) -> tuple[int, int]:
    """Return (ih, iw) scaled so the long edge equals INFER_LONG."""
    scale = INFER_LONG / max(fh, fw)
    return int(fh * scale), int(fw * scale)


# ── Pre / post-processing ─────────────────────────────────────────────────────

def preprocess(frame_bgr: np.ndarray, ih: int, iw: int) -> np.ndarray:
    """
    BGR frame → AnimeGANv3 input tensor [1, ih, iw, 3].

    Steps:
      1. BGR → RGB   (model was trained on RGB)
      2. Resize to (iw, ih)
      3. float32 normalise to [−1, 1]   i.e. pixel / 127.5 − 1
      4. Add batch dimension → NHWC [1, ih, iw, 3]
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    t   = rgb.astype(np.float32) / 127.5 - 1.0
    return t[np.newaxis]


def postprocess(raw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    AnimeGANv3 output tensor [1, ih, iw, 3] → BGR frame at (out_h, out_w).

    Steps:
      1. Remove batch dim → [ih, iw, 3]
      2. Denormalise [−1, 1] → [0, 255]  i.e. (val + 1) × 127.5
      3. Clip and cast to uint8
      4. RGB → BGR
      5. Upscale to original frame resolution (bicubic)
    """
    t   = (raw[0] + 1.0) * 127.5
    t   = np.clip(t, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


# ── Inference thread ──────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Background thread that runs the style model on full camera frames.

    Frame passing uses a "latest-frame" pattern: the main thread always
    overwrites the pending frame with the newest one.  If inference is slower
    than the camera, frames are dropped rather than queued, keeping latency
    low and memory usage constant.
    """

    def __init__(self, sess: ort.InferenceSession) -> None:
        super().__init__(daemon=True)
        self._sess     = sess
        self._inp_name = sess.get_inputs()[0].name

        # Input slot — main thread writes, worker thread reads
        self._in_lock  = threading.Lock()
        self._in_frame: np.ndarray | None = None
        self._in_seq   = 0        # incremented on every push()
        self._proc_seq = -1       # last sequence number processed

        # Output slot — worker thread writes, main thread reads
        self._out_lock  = threading.Lock()
        self._out_frame: np.ndarray | None = None

        self.inf_fps  = 0.0       # updated after each inference pass
        self._running = threading.Event()
        self._running.set()
        self._wakeup  = threading.Event()   # signalled by push()

    def push(self, frame: np.ndarray) -> None:
        """Hand the latest camera frame to the inference thread."""
        with self._in_lock:
            self._in_frame = frame
            self._in_seq  += 1
        self._wakeup.set()

    def get_result(self) -> np.ndarray | None:
        """Return the most recently stylised frame, or None if not ready yet."""
        with self._out_lock:
            return self._out_frame

    def stop(self) -> None:
        """Signal the thread to exit its run loop."""
        self._running.clear()
        self._wakeup.set()

    def run(self) -> None:
        t_prev = time.perf_counter()
        while self._running.is_set():
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()

            # Grab the latest frame; skip if nothing new since last pass
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

            # Blend stylised result with original for a softer transition
            result = cv2.addWeighted(styled, BLEND_ALPHA, frame, 1 - BLEND_ALPHA, 0)

            with self._out_lock:
                self._out_frame = result

            t_now      = time.perf_counter()
            self.inf_fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev     = t_now


# ── On-screen display ─────────────────────────────────────────────────────────

def draw_osd(img: np.ndarray, d_fps: float, i_fps: float,
             mirrored: bool) -> np.ndarray:
    """Overlay FPS counters and model name on a semi-transparent bottom bar."""
    out  = img.copy()
    h, w = out.shape[:2]

    # Semi-transparent black bar at the bottom
    ov = out.copy()
    cv2.rectangle(ov, (0, h - 46), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.4, out, 0.6, 0, out)

    label = "Shinkai" + ("  [mirror]" if mirrored else "")
    cv2.putText(out, label, (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 210, 210), 2, cv2.LINE_AA)

    backend = "OpenVINO" if _OPENVINO else "CPU"
    cv2.putText(out,
                f"disp {d_fps:.0f}  infer {i_fps:.0f} fps  [{backend}]",
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 80), 2, cv2.LINE_AA)
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

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
