#!/usr/bin/env python3
"""Windows webcam/IP-camera version of SMARTIR. Run with no arguments for the GUI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from detector import find_pc_spot as find_spot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from smartir_v2 import (PulseTracker, fit_calibration,
                        load_calibration, map_point, save_calibration,
                        target_positions)

DEFAULT_CAL = HERE / "pc-calibration.json"


def gray_bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def screen_size():
    if platform.system() == "Windows":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except OSError:
            pass
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    result = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
    return result


def open_webcam(index, width, height, fps, exposure=None, gain=None):
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif platform.system() == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    failures = []
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            failures.append(f"backend {backend}: cannot open")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if exposure is not None:
            # Webcam exposure units are driver-specific. 0.25 is the Windows
            # DirectShow manual-exposure convention, not a universal value.
            if backend == cv2.CAP_DSHOW:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, .25)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)
        deadline = time.monotonic() + 2.0
        frame = None
        while time.monotonic() < deadline:
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
                break
        if frame is not None:
            return cap, frame, backend
        cap.release()
        failures.append(f"backend {backend}: no frames")
    raise RuntimeError(f"Webcam {index} unavailable ({'; '.join(failures)}). Close other camera apps or try --camera 1.")


def validate_stream_url(url):
    """Accept only live network camera URLs, never local files or paths."""
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme.lower() in ("http", "https", "rtsp") and parsed.hostname
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("IP camera URL must be an http://, https://, or rtsp:// video stream")
    return url


def open_ip_stream(url, timeout_ms=5000):
    validate_stream_url(url)
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except cv2.error:
        # Older OpenCV builds may lack the Python constructor with params.
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("IP camera stream could not open. Check its video URL, Wi-Fi, and camera app.")
    try:
        ok, frame = cap.read()
    except cv2.error as exc:
        cap.release()
        raise RuntimeError("IP camera connected but could not decode video frames") from exc
    if not ok or frame is None or not frame.size:
        cap.release()
        raise RuntimeError("IP camera connected but returned no video frames. Use a live MJPEG or RTSP video URL, not a snapshot page.")
    return cap, frame, int(cap.get(cv2.CAP_PROP_BACKEND))


def open_camera(args):
    if args.stream_url:
        return open_ip_stream(args.stream_url, args.stream_timeout_ms)
    return open_webcam(args.camera, args.width, args.height,
                       args.fps, args.exposure, args.gain)


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or not frame.size:
        raise RuntimeError("Camera stopped delivering frames; check its connection and restart the mode")
    return frame


class WindowsMouse:
    """Single left-button mouse output, using the supported SendInput API."""

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self, user32=None):
        if user32 is None and platform.system() != "Windows":
            raise RuntimeError("Desktop mouse output is supported on Windows only")
        self.user32 = user32 or ctypes.windll.user32
        self.down = False

    def _button(self, flags):
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]

        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MouseInput)]

        item = Input(0, MouseInput(0, 0, 0, flags, 0, 0))
        if self.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(item)) != 1:
            raise RuntimeError("Windows rejected the mouse event; check desktop permissions")

    def emit(self, event, point):
        if event in ("down", "move") and point is not None:
            if not self.user32.SetCursorPos(round(point[0]), round(point[1])):
                raise RuntimeError("Windows rejected the cursor position")
            if event == "down":
                self._button(self.MOUSEEVENTF_LEFTDOWN)
                self.down = True
        elif event == "up" and self.down:
            self._button(self.MOUSEEVENTF_LEFTUP)
            self.down = False

    def close(self):
        self.emit("up", None)


def camera_controls(args):
    if args.stream_url:
        return {"stream_sha256": hashlib.sha256(args.stream_url.encode("utf-8")).hexdigest()}
    return {"camera_index": args.camera, "requested_width": args.width,
            "requested_height": args.height, "fps": args.fps,
            "exposure": args.exposure, "gain": args.gain}


def view(name, fullscreen=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw_label(image, text, y=30):
    cv2.putText(image, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (255, 255, 255), 2, cv2.LINE_AA)


class CalibrationHold:
    """Require repeated detections at one position, allowing remote flicker."""

    def __init__(self, required=8, tolerance=18, gap_seconds=.4):
        self.required = required
        self.tolerance = tolerance
        self.gap_seconds = gap_seconds
        self.samples = []
        self.last_seen = None

    @property
    def progress(self):
        return min(1.0, len(self.samples) / self.required)

    @property
    def point(self):
        return tuple(np.median(self.samples, axis=0)) if self.samples else None

    def update(self, spot, now):
        if self.last_seen is not None and now - self.last_seen > self.gap_seconds:
            self.samples.clear()
            self.last_seen = None
        if spot is None:
            return False
        if self.samples and np.linalg.norm(np.asarray(spot) - np.asarray(self.point)) > self.tolerance:
            self.samples.clear()
        self.samples.append(spot)
        self.last_seen = now
        return len(self.samples) >= self.required


def annotated_camera(frame, spot, reason, held=None):
    image = frame.copy()
    shown = spot if spot is not None else held
    if shown is not None:
        center = tuple(round(v) for v in shown)
        color = (0, 255, 0) if spot is not None else (0, 190, 255)
        cv2.circle(image, center, 14, color, 2, cv2.LINE_AA)
        cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 10, 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 37), (20, 20, 20), -1)
    label = "LIGHT DETECTED" if spot is not None else ("SIGNAL HELD" if held is not None else reason[:48])
    draw_label(image, label, 26)
    return image


def camera_inset(canvas, frame, spot, reason, avoid=None, held=None):
    """Draw the live camera at a corner away from the current target."""
    height, width = canvas.shape[:2]
    scale = min(width * .28 / frame.shape[1], height * .28 / frame.shape[0], 1.0)
    inset_width = max(1, round(frame.shape[1] * scale))
    inset_height = max(1, round(frame.shape[0] * scale))
    image = cv2.resize(annotated_camera(frame, spot, reason, held=held),
                       (inset_width, inset_height), interpolation=cv2.INTER_AREA)
    pad = 18
    corners = [(pad, pad), (width - inset_width - pad, pad),
               (pad, height - inset_height - pad),
               (width - inset_width - pad, height - inset_height - pad)]
    if avoid is not None:
        x, y = max(corners, key=lambda p: (p[0] + inset_width/2 - avoid[0])**2 +
                                              (p[1] + inset_height/2 - avoid[1])**2)
    else:
        x, y = corners[1]
    canvas[y:y+inset_height, x:x+inset_width] = image
    cv2.rectangle(canvas, (x-2, y-2), (x+inset_width+1, y+inset_height+1),
                  (0, 190, 255), 2)
    return canvas


def calibration_screen(display, target, index, frame, spot, reason, hold, phase):
    canvas = np.zeros((display[1], display[0], 3), np.uint8)
    center = tuple(round(v) for v in target)
    cv2.drawMarker(canvas, center, (0, 180, 255), cv2.MARKER_CROSS, 42, 2)
    cv2.circle(canvas, center, 37, (70, 70, 70), 4, cv2.LINE_AA)
    if hold.progress:
        cv2.ellipse(canvas, center, (37, 37), -90, 0, round(360*hold.progress),
                    (0, 255, 0), 5, cv2.LINE_AA)
    draw_label(canvas, f"Target {index+1}/9: aim remote at crosshair and hold its button")
    draw_label(canvas, "Release remote between targets  |  Esc abort  |  Space skip optional point", 66)
    status = ("Release remote: sampling background" if phase != "capture" else
              f"{'Tracking light' if spot is not None else 'Waiting for light'}: "
              f"{round(100*hold.progress)}%")
    draw_label(canvas, status, 105)
    return camera_inset(canvas, frame, spot, reason, avoid=target,
                        held=hold.point if spot is None else None)


def doctor(args):
    cap, frame, backend = open_camera(args)
    try:
        source = "IP camera" if args.stream_url else f"Webcam {args.camera}"
        print(f"{source}: OK, backend {backend}, frame {frame.shape[1]}x{frame.shape[0]}")
        print(f"Brightness range: {gray_bgr(frame).min()}..{gray_bgr(frame).max()}")
        print(f"Actual FPS setting reported: {cap.get(cv2.CAP_PROP_FPS):.1f}")
        if args.exposure is not None and not args.stream_url:
            print(f"Actual exposure setting reported: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print("The camera must face the whole display before calibration.")
    finally:
        cap.release()


def preview(args):
    cap, frame, _ = open_camera(args)
    name = "SMARTIR PC camera preview - Q to quit"
    try:
        view(name)
        baseline = [gray_bgr(frame)]
        baseline.extend(gray_bgr(read_frame(cap)) for _ in range(11))
        background = np.median(baseline, axis=0).astype(np.uint8)
        camera_tracker = PulseTracker(release_ms=args.release_ms,
                                      max_speed=6000, alpha=.6, deadband=.5)
        while True:
            frame = read_frame(cap)
            current = gray_bgr(frame)
            spot, mask, reason = find_spot(current, background,
                                           min_delta=args.min_delta,
                                           max_area=args.max_area,
                                           preferred=camera_tracker.point if camera_tracker.active else None)
            _, tracked = camera_tracker.update(spot, time.monotonic())
            annotated = annotated_camera(frame, spot, reason,
                                         held=tracked if spot is None else None)
            draw_label(annotated, f"Sensitivity threshold: {args.min_delta} (lower if LED is dim)", 62)
            cv2.imshow(name, annotated)
            stable = mask == 0
            background[stable] = np.uint8(.97*background[stable] + .03*current[stable])
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate(args):
    display = screen_size()
    cap, frame, _ = open_camera(args)
    capture_size = (frame.shape[1], frame.shape[0])
    pairs = []
    name = "SMARTIR PC calibration"
    try:
        view(name, fullscreen=True)
        targets = target_positions(*display)
        for index, target in enumerate(targets):
            baseline = []
            hold = CalibrationHold()
            started = time.monotonic()
            phase = "settle"
            captured = False
            while time.monotonic()-started < args.target_timeout:
                frame = read_frame(cap)
                current = gray_bgr(frame)
                elapsed = time.monotonic()-started
                spot, reason = None, "waiting for light"
                if phase == "settle" and elapsed > .7:
                    phase = "baseline"
                if phase == "baseline":
                    baseline.append(current.copy())
                    if len(baseline) >= 8:
                        background = np.median(baseline, axis=0).astype(np.uint8)
                        phase = "capture"
                elif phase == "capture":
                    spot, _, reason = find_spot(current, background,
                                                min_delta=args.min_delta,
                                                max_area=args.max_area,
                                                preferred=hold.point)
                    captured = hold.update(spot, time.monotonic())
                status = calibration_screen(display, target, index, frame, spot,
                                            reason, hold, phase)
                cv2.imshow(name, status)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt()
                if key == 32 and index >= 4:
                    break
                if captured:
                    pairs.append((hold.point, target))
                    confirmation = calibration_screen(display, target, index, frame,
                                                      spot, "captured", hold, "capture")
                    cv2.putText(confirmation, "POINT CALIBRATED", (30, display[1]-45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow(name, confirmation)
                    cv2.waitKey(450)
                    break
            print(f"Target {index+1}: {'captured' if captured else 'missed'}")
        profile = fit_calibration(pairs, capture_size, display)
        profile["source"] = "pc-ipcam" if args.stream_url else "pc-webcam"
        profile["camera_controls"] = camera_controls(args)
        save_calibration(profile, args.calibration)
        print(f"Saved {args.calibration}; {len(pairs)}/9 targets; "
              f"mean {profile['mean_error_px']:.1f}px, max {profile['max_error_px']:.1f}px")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run(args):
    display = screen_size()
    cap, frame, _ = open_camera(args)
    capture_size = (frame.shape[1], frame.shape[0])
    output = None
    try:
        H, profile = load_calibration(args.calibration, capture_size, display)
        expected_source = "pc-ipcam" if args.stream_url else "pc-webcam"
        if profile.get("source") != expected_source or profile.get("camera_controls") != camera_controls(args):
            raise RuntimeError("Camera settings differ from PC calibration; recalibrate or use the same options")
        tracker = PulseTracker(release_ms=args.release_ms)
        camera_tracker = PulseTracker(release_ms=args.release_ms,
                                      max_speed=6000, alpha=.6, deadband=.5)
        canvas = np.zeros((display[1], display[0], 3), np.uint8) if args.output == "canvas" else None
        if args.output == "mouse":
            print("WARNING: Without an IR-pass filter, changing visible light can cause false mouse clicks.")
            output = WindowsMouse()
        if canvas is not None:
            view("SMARTIR PC whiteboard", fullscreen=True)
            cv2.imshow("SMARTIR PC whiteboard", canvas)
            cv2.waitKey(250)
        else:
            view("SMARTIR PC tracking preview - Q to quit")
        baseline = [gray_bgr(read_frame(cap)) for _ in range(12)]
        background = np.median(baseline, axis=0).astype(np.uint8)
        previous = None
        while True:
            frame = read_frame(cap)
            current = gray_bgr(frame)
            spot, mask, reason = find_spot(current, background,
                                           min_delta=args.min_delta,
                                           max_area=args.max_area,
                                           preferred=camera_tracker.point if camera_tracker.active else None)
            _, camera_point = camera_tracker.update(spot, time.monotonic())
            point = map_point(H, spot, display) if spot is not None else None
            event, smoothed = tracker.update(point, time.monotonic())
            if output is not None:
                output.emit(event, smoothed)
                annotated = annotated_camera(frame, spot, reason,
                                             held=camera_point if spot is None else None)
                draw_label(annotated, f"Mouse: {event}", 62)
                cv2.imshow("SMARTIR PC tracking preview - Q to quit", annotated)
            else:
                if event == "down" and smoothed is not None:
                    previous = tuple(round(v) for v in smoothed)
                    cv2.circle(canvas, previous, 4, (255, 255, 255), -1)
                elif event == "move" and smoothed is not None and previous is not None:
                    point_px = tuple(round(v) for v in smoothed)
                    cv2.line(canvas, previous, point_px, (255, 255, 255), 4, cv2.LINE_AA)
                    previous = point_px
                elif event == "up":
                    previous = None
                visible = camera_inset(canvas.copy(), frame, spot, reason,
                                       held=camera_point if spot is None else None)
                cv2.imshow("SMARTIR PC whiteboard", visible)
            if not tracker.active:
                stable = mask == 0
                background[stable] = np.uint8(.97*background[stable] + .03*current[stable])
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c") and canvas is not None:
                canvas[:] = 0
    finally:
        try:
            if output is not None:
                output.close()
        finally:
            cap.release()
            cv2.destroyAllWindows()


def gui(args):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("SMARTIR PC camera")
    root.geometry("540x420")
    tk.Label(root, text="SMARTIR · PC camera test", font=("Segoe UI", 17, "bold")).pack(pady=12)
    tk.Label(root, text="Point the USB or IP camera at the entire laptop screen.\n"
                        "The IR emitter must face the camera at each crosshair.").pack(pady=5)
    controls = tk.Frame(root)
    controls.pack(pady=8)
    tk.Label(controls, text="Webcam index").grid(row=0, column=0, padx=6)
    camera_index = tk.StringVar(value=str(args.camera))
    tk.Spinbox(controls, from_=0, to=8, textvariable=camera_index, width=5).grid(row=0, column=1)
    tk.Label(controls, text="Brightness threshold").grid(row=0, column=2, padx=6)
    threshold = tk.StringVar(value=str(args.min_delta))
    tk.Spinbox(controls, from_=5, to=200, textvariable=threshold, width=5).grid(row=0, column=3)
    tk.Label(root, text="IP camera video URL (optional; overrides webcam index)").pack(pady=(7, 1))
    stream_url = tk.StringVar(value=args.stream_url or "")
    tk.Entry(root, textvariable=stream_url, width=66).pack(pady=2)
    tk.Label(root, text="Examples: http://192.168.1.5:8080/video  or  rtsp://192.168.1.5/live",
             fg="#555555").pack(pady=(0, 8))
    process = None

    def poll_child():
        if process is not None and process.poll() is None:
            root.after(300, poll_child)
        else:
            root.deiconify()
            if process is not None and process.returncode:
                _, errors = process.communicate()
                messagebox.showerror("SMARTIR camera error", errors.strip() or
                                     f"Mode exited with code {process.returncode}. "
                                     "Check that the webcam is connected and not open elsewhere.")

    def launch(command, output=None):
        nonlocal process
        if process is not None and process.poll() is None:
            messagebox.showwarning("SMARTIR", "Stop the current mode first")
            return
        try:
            camera = int(camera_index.get())
            delta = int(threshold.get())
            if camera < 0 or not 1 <= delta <= 255:
                raise ValueError()
        except ValueError:
            messagebox.showerror("SMARTIR", "Camera index and threshold must be valid numbers")
            return
        url = stream_url.get().strip()
        if url:
            try:
                validate_stream_url(url)
            except ValueError as exc:
                messagebox.showerror("SMARTIR", str(exc))
                return
        if output == "mouse" and not messagebox.askyesno(
            "Enable desktop mouse?", "Visible-light changes may cause false clicks without an IR-pass filter. Continue?"
        ):
            return
        command_line = [sys.executable, str(Path(__file__).resolve()), command,
                        "--camera", str(camera), "--min-delta", str(delta),
                        "--calibration", str(args.calibration),
                        "--width", str(args.width), "--height", str(args.height),
                        "--fps", str(args.fps), "--max-area", str(args.max_area),
                        "--release-ms", str(args.release_ms),
                        "--target-timeout", str(args.target_timeout),
                        "--stream-timeout-ms", str(args.stream_timeout_ms)]
        if args.exposure is not None:
            command_line += ["--exposure", str(args.exposure)]
        if args.gain is not None:
            command_line += ["--gain", str(args.gain)]
        if output is not None:
            command_line += ["--output", output]
        child_env = os.environ.copy()
        if url:
            child_env["SMARTIR_STREAM_URL"] = url
        else:
            child_env.pop("SMARTIR_STREAM_URL", None)
        process = subprocess.Popen(command_line, stderr=subprocess.PIPE,
                                   text=True, env=child_env)
        root.withdraw()
        root.after(300, poll_child)

    for label, command, output in (
        ("1  Camera preview", "preview", None),
        ("2  Calibrate screen", "calibrate", None),
        ("3  Test whiteboard", "run", "canvas"),
        ("4  Control desktop mouse", "run", "mouse"),
    ):
        tk.Button(root, text=label, width=31,
                  command=lambda c=command, o=output: launch(c, o)).pack(pady=4)
    tk.Label(root, text="Q / Esc closes a mode · C clears the whiteboard", fg="#74420e").pack(pady=8)
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="gui",
                        choices=("gui", "doctor", "preview", "calibrate", "run"))
    parser.add_argument("--camera", type=int, default=0, help="Webcam index; try 1 for an external USB webcam")
    parser.add_argument("--stream-url", default=os.environ.get("SMARTIR_STREAM_URL", ""),
                        help="Optional HTTP(S) MJPEG or RTSP IP camera video URL; overrides --camera")
    parser.add_argument("--stream-timeout-ms", type=int, default=5000,
                        help="IP camera open/read timeout in milliseconds")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure", type=float, help="Optional backend-specific manual exposure value")
    parser.add_argument("--gain", type=float, help="Optional backend-specific camera gain")
    parser.add_argument("--min-delta", type=int, default=18)
    parser.add_argument("--max-area", type=int, default=250)
    parser.add_argument("--release-ms", type=int, default=300)
    parser.add_argument("--target-timeout", type=float, default=8.0)
    parser.add_argument("--no-auto-test", action="store_true",
                        help="Stop after calibration instead of opening the blank test canvas")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--output", choices=("canvas", "mouse"), default="canvas")
    args = parser.parse_args(argv)
    if args.camera < 0 or min(args.width, args.height) < 160 or args.fps < 1 or args.min_delta < 1 or args.max_area < 2 or args.stream_timeout_ms < 100:
        parser.error("Invalid camera or detector setting")
    if args.stream_url:
        try:
            validate_stream_url(args.stream_url)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        if args.command == "doctor": doctor(args)
        elif args.command == "preview": preview(args)
        elif args.command == "calibrate":
            calibrate(args)
            if not args.no_auto_test:
                print("Calibration saved; opening the test whiteboard. Press C to clear, Q to quit.")
                args.output = "canvas"
                run(args)
        elif args.command == "run": run(args)
        else: gui(args)
    except KeyboardInterrupt:
        print("Stopped")
    except (RuntimeError, OSError, ValueError, cv2.error) as exc:
        print(f"SMARTIR PC error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
