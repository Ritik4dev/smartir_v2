#!/usr/bin/env python3
"""Windows webcam version of SMARTIR. Run with no arguments for the GUI."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from smartir_v2 import (PulseTracker, find_spot, fit_calibration,
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


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or not frame.size:
        raise RuntimeError("Webcam stopped delivering frames; close other apps and reconnect it")
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


def doctor(args):
    cap, frame, backend = open_webcam(args.camera, args.width, args.height,
                                      args.fps, args.exposure, args.gain)
    try:
        print(f"Webcam {args.camera}: OK, backend {backend}, frame {frame.shape[1]}x{frame.shape[0]}")
        print(f"Brightness range: {gray_bgr(frame).min()}..{gray_bgr(frame).max()}")
        print(f"Actual FPS setting reported: {cap.get(cv2.CAP_PROP_FPS):.1f}")
        if args.exposure is not None:
            print(f"Actual exposure setting reported: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print("A laptop's built-in webcam may not see its own screen plane; an external webcam must face the whole display.")
    finally:
        cap.release()


def preview(args):
    cap, frame, _ = open_webcam(args.camera, args.width, args.height,
                                args.fps, args.exposure, args.gain)
    name = "SMARTIR PC camera preview - Q to quit"
    try:
        view(name)
        baseline = [gray_bgr(frame)]
        baseline.extend(gray_bgr(read_frame(cap)) for _ in range(11))
        background = np.median(baseline, axis=0).astype(np.uint8)
        while True:
            frame = read_frame(cap)
            current = gray_bgr(frame)
            spot, mask, reason = find_spot(current, background,
                                           min_delta=args.min_delta,
                                           max_area=args.max_area)
            if spot is not None:
                cv2.circle(frame, tuple(round(v) for v in spot), 9, (0, 255, 0), 2)
            draw_label(frame, reason)
            cv2.imshow(name, frame)
            stable = mask == 0
            background[stable] = np.uint8(.97*background[stable] + .03*current[stable])
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate(args):
    display = screen_size()
    cap, frame, _ = open_webcam(args.camera, args.width, args.height,
                                args.fps, args.exposure, args.gain)
    capture_size = (frame.shape[1], frame.shape[0])
    pairs = []
    name = "SMARTIR PC calibration"
    try:
        view(name, fullscreen=True)
        targets = target_positions(*display)
        for index, target in enumerate(targets):
            canvas = np.zeros((display[1], display[0], 3), np.uint8)
            cv2.drawMarker(canvas, tuple(round(v) for v in target),
                           (0, 180, 255), cv2.MARKER_CROSS, 42, 2)
            draw_label(canvas, f"Target {index+1}/9: release remote, then press LED at crosshair")
            draw_label(canvas, "First four corners required | ESC abort | SPACE skip extra point", 66)
            cv2.imshow(name, canvas)
            cv2.waitKey(20)
            baseline, samples = [], []
            started = time.monotonic()
            phase = "settle"
            while time.monotonic()-started < args.target_timeout:
                frame = read_frame(cap)
                current = gray_bgr(frame)
                elapsed = time.monotonic()-started
                if phase == "settle" and elapsed > .7:
                    phase = "baseline"
                if phase == "baseline":
                    baseline.append(current.copy())
                    if len(baseline) >= 8:
                        background = np.median(baseline, axis=0).astype(np.uint8)
                        phase = "capture"
                elif phase == "capture":
                    spot, _, _ = find_spot(current, background,
                                           min_delta=args.min_delta,
                                           max_area=args.max_area)
                    if spot is not None and (not samples or
                        np.linalg.norm(np.asarray(spot)-np.median(samples, axis=0)) < 18):
                        samples.append(spot)
                    if len(samples) >= 3:
                        pairs.append((tuple(np.median(samples, axis=0)), target))
                        break
                status = canvas.copy()
                draw_label(status, f"{phase}: {len(samples)}/3 camera hits", 105)
                cv2.imshow(name, status)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt()
                if key == 32:
                    break
            print(f"Target {index+1}: {'captured' if len(samples)>=3 else 'missed'}")
        profile = fit_calibration(pairs, capture_size, display)
        profile["source"] = "pc-webcam"
        profile["camera_controls"] = camera_controls(args)
        save_calibration(profile, args.calibration)
        print(f"Saved {args.calibration}; {len(pairs)}/9 targets; "
              f"mean {profile['mean_error_px']:.1f}px, max {profile['max_error_px']:.1f}px")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run(args):
    display = screen_size()
    cap, frame, _ = open_webcam(args.camera, args.width, args.height,
                                args.fps, args.exposure, args.gain)
    capture_size = (frame.shape[1], frame.shape[0])
    output = None
    try:
        H, profile = load_calibration(args.calibration, capture_size, display)
        if profile.get("source") != "pc-webcam" or profile.get("camera_controls") != camera_controls(args):
            raise RuntimeError("Camera settings differ from PC calibration; recalibrate or use the same options")
        tracker = PulseTracker(release_ms=args.release_ms)
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
                                           max_area=args.max_area)
            point = map_point(H, spot, display) if spot is not None else None
            event, smoothed = tracker.update(point, time.monotonic())
            if output is not None:
                output.emit(event, smoothed)
                if spot is not None:
                    cv2.circle(frame, tuple(round(v) for v in spot), 9, (0, 255, 0), 2)
                draw_label(frame, f"{reason} | {event}")
                cv2.imshow("SMARTIR PC tracking preview - Q to quit", frame)
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
                cv2.imshow("SMARTIR PC whiteboard", canvas)
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
    root.title("SMARTIR PC webcam")
    root.geometry("480x340")
    tk.Label(root, text="SMARTIR · PC webcam test", font=("Segoe UI", 17, "bold")).pack(pady=12)
    tk.Label(root, text="Place an external webcam where it can see the entire laptop screen.\n"
                        "The IR emitter must face the webcam at each crosshair.").pack(pady=5)
    controls = tk.Frame(root)
    controls.pack(pady=8)
    tk.Label(controls, text="Webcam index").grid(row=0, column=0, padx=6)
    camera_index = tk.StringVar(value=str(args.camera))
    tk.Spinbox(controls, from_=0, to=8, textvariable=camera_index, width=5).grid(row=0, column=1)
    tk.Label(controls, text="Brightness threshold").grid(row=0, column=2, padx=6)
    threshold = tk.StringVar(value=str(args.min_delta))
    tk.Spinbox(controls, from_=5, to=200, textvariable=threshold, width=5).grid(row=0, column=3)
    process = None

    def poll_child():
        if process is not None and process.poll() is None:
            root.after(300, poll_child)
        else:
            root.deiconify()

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
                        "--target-timeout", str(args.target_timeout)]
        if args.exposure is not None:
            command_line += ["--exposure", str(args.exposure)]
        if args.gain is not None:
            command_line += ["--gain", str(args.gain)]
        if output is not None:
            command_line += ["--output", output]
        process = subprocess.Popen(command_line)
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
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure", type=float, help="Optional backend-specific manual exposure value")
    parser.add_argument("--gain", type=float, help="Optional backend-specific camera gain")
    parser.add_argument("--min-delta", type=int, default=28)
    parser.add_argument("--max-area", type=int, default=250)
    parser.add_argument("--release-ms", type=int, default=170)
    parser.add_argument("--target-timeout", type=float, default=8.0)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--output", choices=("canvas", "mouse"), default="canvas")
    args = parser.parse_args(argv)
    if args.camera < 0 or min(args.width, args.height) < 160 or args.fps < 1 or args.min_delta < 1 or args.max_area < 2:
        parser.error("Invalid camera or detector setting")
    try:
        if args.command == "doctor": doctor(args)
        elif args.command == "preview": preview(args)
        elif args.command == "calibrate": calibrate(args)
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
