# 🎌 AnimeCam

> **Real-time Shinkai-style neural webcam** — point your camera at yourself, stream anime.

Powered by **AnimeGANv3 Shinkai** · ONNX Runtime · OpenVINO · Python

---

## ✨ What it does

Your webcam feed → scaled down → AnimeGANv3 Shinkai model → scaled back up → *you, but animated*.

```
Camera 1280×720
    │
    ▼  scale to 256×144
InferenceThread  ──→  AnimeGANv3 Shinkai  ──→  upscale to 1280×720
    │
    ▼
Display  (always 30 fps, style updates ~14 fps)
```

The display thread and inference thread run independently — the window is always smooth even when the model is thinking.

---

## 🚀 Quick start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install opencv-python onnxruntime onnxruntime-openvino numpy

# 3. Run
python main.py
```

**Controls**

| Key | Action |
|-----|--------|
| `M` | Mirror flip |
| `ESC` | Quit |

---

## ⚙️ Configuration

All knobs are at the top of `main.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `INFER_LONG` | `256` | Long-edge inference resolution. Raise to `384`/`512` for sharper style, costs FPS. |
| `BLEND_ALPHA` | `0.95` | `1.0` = pure Shinkai · `0.0` = original feed |
| `CAMERA_ID` | `0` | Camera device index |

---

## 🧠 Model

**AnimeGANv3 Shinkai** — trained to reproduce the painterly look of Makoto Shinkai's films (*Your Name*, *Weathering With You*).

- Format: ONNX, NHWC layout
- Input: `[1, H, W, 3]` RGB normalised to `[−1, 1]`
- Output: `[1, H, W, 3]` same range, same dims
- Dynamic spatial dims — accepts any resolution
- Size: **4.1 MB**
- Source: [TachibanaYoshino/AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3)

---

## 📊 Performance

| Backend | Inference @ 256×144 |
|---------|-------------------|
| CPU (ORT) | ~8 fps |
| OpenVINO CPU | ~14 fps |
| Display thread | always 30 fps |

---

## 🎥 TODO — Virtual camera output

Stream the stylised feed directly into OBS, Discord, Zoom, or any app that accepts a webcam source via **v4l2loopback**.

### 1 — Load the kernel module

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="AnimeCam" exclusive_caps=1

# Persist across reboots
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="AnimeCam" exclusive_caps=1' \
    | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### 2 — Install

```bash
# Arch
yay -S v4l2loopback-dkms

# Ubuntu / Debian
sudo apt install v4l2loopback-dkms
```

### 3 — Implementation outline (in `main.py`)

```python
import fcntl, ctypes, os

V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE            = 1
V4L2_PIX_FMT_BGR24         = 0x33524742

class v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width",        ctypes.c_uint32),
        ("height",       ctypes.c_uint32),
        ("pixelformat",  ctypes.c_uint32),
        ("field",        ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage",    ctypes.c_uint32),
        ("colorspace",   ctypes.c_uint32),
        ("priv",         ctypes.c_uint32),
    ]

class v4l2_format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", v4l2_pix_format)]

VIDIOC_S_FMT = 0xC0D05605

def open_vcam(device="/dev/video10", width=1280, height=720):
    fd = open(device, "wb", buffering=0)
    fmt               = v4l2_format()
    fmt.type          = V4L2_BUF_TYPE_VIDEO_OUTPUT
    fmt.fmt.width     = width
    fmt.fmt.height    = height
    fmt.fmt.pixelformat  = V4L2_PIX_FMT_BGR24
    fmt.fmt.field        = V4L2_FIELD_NONE
    fmt.fmt.bytesperline = width * 3
    fmt.fmt.sizeimage    = width * height * 3
    fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
    return fd

def write_vcam(fd, frame_bgr: np.ndarray) -> None:
    buf = frame_bgr if frame_bgr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_bgr)
    fd.write(buf.tobytes())
```

### 4 — Integration point

Call `write_vcam(fd, result)` inside `InferenceThread.run()` after updating `self._out_frame`, or spin up a dedicated third thread that reads `get_result()` and writes to the device at display FPS.

---

## 🗺️ Roadmap

### 🔲 Native C++ single-binary OBS source

The long-term goal is a **self-contained C++ application** that replaces the Python prototype entirely — no interpreter, no venv, one binary you drop anywhere and run.

**Why C++**
- Single statically-linked executable: copy to any Linux machine and it just works
- No Python runtime, no pip, no virtual environment management
- Lower latency: direct memory path from camera → model → v4l2 device
- Easier to package as an OBS plugin or systemd service
- Full control over threading, memory layout, and buffer lifetimes

**Planned architecture**

```
┌─────────────────────────────────────────────────────────┐
│  animecam  (single binary)                              │
│                                                         │
│  Thread 1 — CameraCapture                               │
│    V4L2 → mmap capture → raw BGR frames                 │
│    → LatestFrame<cv::Mat>  (lock-free slot, drop-old)   │
│                                                         │
│  Thread 2 — Inference (ncnn-vulkan or ORT)              │
│    BGR frame → resize → Shinkai model                   │
│    → stylised BGR frame                                 │
│    → LatestFrame<cv::Mat>                               │
│                                                         │
│  Thread 3 — VirtualCam output                           │
│    stylised frame → VIDIOC_S_FMT → write() → /dev/videoX│
│    OBS / Discord / Zoom sees it as a regular webcam     │
└─────────────────────────────────────────────────────────┘
```

**Key components to implement**

| Component | Technology |
|-----------|-----------|
| Camera capture | V4L2 `mmap` buffers (zero-copy) |
| Inference backend | ncnn + Vulkan EP (AMD iGPU, 30+ fps) |
| Model format | ONNX → ncnn `param`/`bin` (FP16) |
| Virtual camera | v4l2loopback · `VIDIOC_S_FMT` · raw `write()` |
| Build system | CMake · static linking where possible |
| CLI | `--device`, `--vcam`, `--infer-size`, `--blend` flags |

**Static binary checklist**

```cmake
# Link ncnn statically
set(NCNN_BUILD_SHARED_LIBS OFF)

# Link OpenCV statically (or use minimal subset)
set(BUILD_SHARED_LIBS OFF)

# Strip and compress final binary
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
```

**OBS integration path**

```
Option A — Virtual webcam (simplest)
  animecam writes to /dev/video10  →  OBS adds it as "Video Capture Device"

Option B — OBS plugin (advanced)
  Implement obs_source_t with get_frame() callback
  animecam becomes a native OBS source plugin (.so)
  Users install it from OBS → Tools → Scripts or plugin folder
```

**Estimated milestone order**

1. Port Python pipeline to C++ with OpenCV + ncnn-vulkan (no virtual cam yet)
2. Add v4l2loopback write — verify OBS sees the feed
3. Replace Haar face detection with YuNet (OpenCV DNN, no extra deps)
4. Static link and strip binary — verify it runs without any system libs beyond glibc
5. Package as single `.tar.gz` with install script for modprobe persistence
6. (optional) OBS native plugin wrapper

---

## 📁 Project structure

```
anime-cam/
├── main.py                      # entire pipeline
├── requirements.txt
└── models/
    └── AnimeGANv3_Shinkai.onnx  # 4.1 MB
```

---

## 📦 Dependencies

```
opencv-python
onnxruntime
onnxruntime-openvino   # optional, ~30 % faster on CPU
numpy
```

---

*Built for streamers. Runs on CPU. No GPU required.*
