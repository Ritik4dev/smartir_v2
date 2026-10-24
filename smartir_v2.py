#!/usr/bin/env python3
"""SMARTIR v2: OV5647/TV-remote whiteboard for Raspberry Pi OS.

No module import opens the camera, display, or virtual input device. This
allows diagnostics and unit tests on development machines without Pi hardware.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CAL = APP_DIR / "smartir-calibration.json"


def camera_open(width, height, fps, exposure_us, gain):
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("Picamera2 is missing. Install python3-picamera2 with apt.") from exc
    camera = Picamera2()
    try:
        config = camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
            controls={"FrameDurationLimits": (round(1e6 / fps), round(1e6 / fps))},
            queue=False,
        )
        camera.configure(config)
        camera.set_controls({"AeEnable": False, "AwbEnable": False,
                             "ExposureTime": exposure_us, "AnalogueGain": gain})
        camera.start()
        time.sleep(.4)
        frame = camera.capture_array("main")
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"Camera returned {None if frame is None else frame.shape}; requested {width}x{height}")
        return camera
    except Exception:
        camera.close()
        raise


def gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def find_spot(frame_gray, background, roi=None, min_delta=28, min_peak=45,
              min_area=2, max_area=250, max_spots=3):
    """Find compact, newly brighter light; returns (point, mask, reason).

    This detects temporal change, NOT infrared wavelength. Moving visible
    highlights can still be confused with a remote without an IR-pass filter.
    """
    if frame_gray.shape != background.shape:
        raise ValueError("Background size does not match camera frame")
    delta = cv2.subtract(frame_gray, background)
    threshold = max(min_delta, int(np.percentile(delta, 99.8) * .48))
    mask = np.uint8(delta >= threshold) * 255
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not min_area <= area <= max_area or max(w, h) > 32:
            continue
        region = frame_gray[y:y+h, x:x+w]
        peak = int(region.max())
        contrast = int(delta[y:y+h, x:x+w].max())
        if peak < min_peak or contrast < min_delta:
            continue
        component = labels[y:y+h, x:x+w] == i
        weights = delta[y:y+h, x:x+w].astype(np.float32) * component
        yy, xx = np.indices(weights.shape)
        total = float(weights.sum())
        centroid = (x + float((weights * xx).sum()) / total,
                    y + float((weights * yy).sum()) / total)
        candidates.append((contrast * min(area, 30), centroid))
    if not candidates:
        return None, mask, "no compact bright change"
    if len(candidates) > max_spots:
        return None, mask, "ambiguous: multiple changing lights"
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * .65:
        return None, mask, "ambiguous: two similar lights"
    return candidates[0][1], mask, "spot"


def target_positions(width, height):
    # Four Lee-style corners first, then five points that reveal distortion
    # and reduce the effect of a single imprecise remote placement.
    fractions = [(.10, .10), (.90, .10), (.90, .90), (.10, .90),
                 (.50, .10), (.90, .50), (.50, .90), (.10, .50), (.50, .50)]
    return [(width*x, height*y) for x, y in fractions]


def fit_calibration(pairs, camera_size, screen_size):
    if len(pairs) < 6:
        raise ValueError("Need at least 6 of 9 calibration targets")
    src = np.float32([p[0] for p in pairs])
    dst = np.float32([p[1] for p in pairs])
    for corner in target_positions(*screen_size)[:4]:
        if not any(np.linalg.norm(point - corner) < 2 for point in dst):
            raise ValueError("All four corner targets are required; retry calibration")
    xspan = np.ptp(dst[:, 0]) / screen_size[0]
    yspan = np.ptp(dst[:, 1]) / screen_size[1]
    if min(xspan, yspan) < .70:
        raise ValueError("Targets do not cover enough of the screen")
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if H is None or mask is None or not np.isfinite(H).all():
        raise ValueError("Could not fit a valid homography")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    inliers = mask.ravel().astype(bool)
    diag = float(np.hypot(*screen_size))
    for corner in target_positions(*screen_size)[:4]:
        corner_index = int(np.argmin(np.linalg.norm(dst - corner, axis=1)))
        if not inliers[corner_index]:
            raise ValueError("A corner was rejected as an outlier; retry that calibration")
    if sum(inliers) < 6 or np.max(errors[inliers]) > .012 * diag:
        raise ValueError("Calibration is inaccurate; retry with a fixed camera and contact-point emitter")
    # RANSAC can mask an entire bad target. Do not allow a saved profile
    # which visibly misses a point and will produce unsafe desktop clicks.
    if np.max(errors) > .025 * diag:
        raise ValueError("One or more targets are badly misplaced; retry calibration")
    try:
        inverse = np.linalg.inv(H)
        screen_corners = np.float32([[[0, 0]], [[screen_size[0]-1, 0]],
                                     [[screen_size[0]-1, screen_size[1]-1]],
                                     [[0, screen_size[1]-1]]])
        camera_corners = cv2.perspectiveTransform(screen_corners, inverse).reshape(-1, 2)
    except (np.linalg.LinAlgError, cv2.error) as exc:
        raise ValueError("Calibration transform cannot be inverted") from exc
    if not np.isfinite(camera_corners).all():
        raise ValueError("Screen projects outside the camera view")
    cw, ch = camera_size
    if (np.min(camera_corners[:, 0]) < -.05*cw or
            np.max(camera_corners[:, 0]) > 1.05*cw or
            np.min(camera_corners[:, 1]) < -.05*ch or
            np.max(camera_corners[:, 1]) > 1.05*ch):
        raise ValueError("Camera does not see the full display; widen/re-aim lens and recalibrate")
    return {"version": 2, "camera_size": list(camera_size),
            "screen_size": list(screen_size), "homography": H.tolist(),
            "camera_points": src.tolist(), "screen_points": dst.tolist(),
            "mean_error_px": float(np.mean(errors)), "max_error_px": float(np.max(errors))}


def save_calibration(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_calibration(path, camera_size, screen_size):
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("No valid calibration. Run 'calibrate' first.") from exc
    if profile.get("version") != 2 or profile.get("camera_size") != list(camera_size):
        raise RuntimeError("Camera settings changed; recalibrate.")
    if profile.get("screen_size") != list(screen_size):
        raise RuntimeError("Display resolution changed; recalibrate.")
    H = np.asarray(profile.get("homography"), dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        raise RuntimeError("Calibration matrix is invalid; recalibrate.")
    return H, profile


def map_point(H, point, screen_size, margin=.03):
    mapped = cv2.perspectiveTransform(np.float32([[point]]), H)[0, 0]
    w, h = screen_size
    x, y = float(mapped[0]), float(mapped[1])
    if not np.isfinite([x, y]).all() or not (-w*margin <= x <= w*(1+margin)
                                              and -h*margin <= y <= h*(1+margin)):
        return None
    return (min(w-1, max(0, x)), min(h-1, max(0, y)))


class PulseTracker:
    """Temporal hold for intermittent remote bursts, with jump rejection and smoothing."""
    def __init__(self, release_ms=170, max_speed=3500, alpha=.55, deadband=1.5):
        self.release_s = release_ms / 1000
        self.max_speed = max_speed
        self.alpha = alpha
        self.deadband = deadband
        self.active = False
        self.point = None
        self.last_seen = None

    def update(self, candidate, now):
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=float)
            if self.active:
                elapsed = max(now - self.last_seen, .01)
                if np.linalg.norm(candidate - self.point) > max(65, self.max_speed*elapsed):
                    candidate = None
            if candidate is not None:
                if not self.active:
                    self.point = candidate
                    self.active = True
                    event = "down"
                else:
                    if np.linalg.norm(candidate - self.point) > self.deadband:
                        self.point = self.point + self.alpha*(candidate-self.point)
                    event = "move"
                self.last_seen = now
                return event, tuple(self.point)
        if self.active and now-self.last_seen >= self.release_s:
            self.active = False
            self.point = None
            return "up", None
        return "hold", tuple(self.point) if self.active else None


class TouchOutput:
    def __init__(self, width, height):
        try:
            from evdev import UInput, AbsInfo, ecodes as e
            self.e = e
            self.dev = UInput({e.EV_KEY: [e.BTN_TOUCH],
                               e.EV_ABS: [(e.ABS_X, AbsInfo(0, 0, width-1, 0, 0, 0)),
                                          (e.ABS_Y, AbsInfo(0, 0, height-1, 0, 0, 0))]},
                              name="SMARTIR virtual touchscreen",
                              input_props=[e.INPUT_PROP_DIRECT])
        except Exception as exc:
            raise RuntimeError("Cannot create /dev/uinput touchscreen. Install python3-evdev, load uinput, and grant user device access.") from exc
        self.down = False

    def emit(self, event, point):
        if event in ("down", "move") and point is not None:
            if event == "down":
                self.dev.write(self.e.EV_KEY, self.e.BTN_TOUCH, 1)
                self.down = True
            self.dev.write(self.e.EV_ABS, self.e.ABS_X, round(point[0]))
            self.dev.write(self.e.EV_ABS, self.e.ABS_Y, round(point[1]))
            self.dev.syn()
        elif event == "up" and self.down:
            self.dev.write(self.e.EV_KEY, self.e.BTN_TOUCH, 0)
            self.dev.syn()
            self.down = False

    def close(self):
        self.emit("up", None)
        self.dev.close()


def display_size(override):
    if override:
        try:
            w, h = (int(x) for x in override.lower().split("x"))
            if min(w, h) < 300:
                raise ValueError()
            return w, h
        except ValueError as exc:
            raise RuntimeError("--screen must be WIDTHxHEIGHT, e.g. 1920x1080") from exc
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No graphical desktop. Raspberry Pi OS Lite can run doctor/snapshot, but calibration and touch require a desktop session.")
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    size = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
    return size


def camera_settings(args):
    if args.exposure_us >= 1e6 / args.fps:
        raise RuntimeError("Exposure must be shorter than the frame period; lower --exposure-us or --fps.")
    return args.width, args.height, args.fps, args.exposure_us, args.gain


def cv_window(name, fullscreen=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def calibrate(args):
    size = display_size(args.screen)
    camera = camera_open(*camera_settings(args))
    pairs = []
    try:
        cv_window("SMARTIR calibration", True)
        cv_window("SMARTIR camera preview")
        for index, target in enumerate(target_positions(*size)):
            canvas = np.zeros((size[1], size[0], 3), np.uint8)
            tx, ty = (round(v) for v in target)
            cv2.drawMarker(canvas, (tx, ty), (0, 180, 255), cv2.MARKER_CROSS, 38, 2)
            cv2.putText(canvas, f"Target {index+1}/9: release remote, then press at crosshair",
                        (25, 45), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(canvas, "First 4 corners required | ESC abort | SPACE skip extra target", (25, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (200, 200, 200), 1)
            cv2.imshow("SMARTIR calibration", canvas)
            baseline_frames, samples = [], []
            started = time.monotonic()
            phase = "settle"
            while time.monotonic()-started < args.target_timeout:
                rgb = camera.capture_array("main")
                current = gray(rgb)
                elapsed = time.monotonic()-started
                if phase == "settle" and elapsed > .65:
                    phase = "baseline"
                if phase == "baseline":
                    baseline_frames.append(current.copy())
                    if len(baseline_frames) >= 8:
                        background = np.median(baseline_frames, axis=0).astype(np.uint8)
                        phase = "capture"
                elif phase == "capture":
                    spot, _, reason = find_spot(current, background, min_delta=args.min_delta)
                    if spot is not None:
                        if not samples or np.linalg.norm(np.asarray(spot)-np.median(samples, axis=0)) < 18:
                            samples.append(spot)
                    if len(samples) >= 3:
                        pairs.append((tuple(np.median(samples, axis=0)), target))
                        break
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(preview, f"{phase}: {len(samples)}/3 hits", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
                cv2.imshow("SMARTIR camera preview", preview)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    raise KeyboardInterrupt()
                if key == 32:
                    break
            print(f"Target {index+1}: {'captured' if len(samples)>=3 else 'missed'}")
        profile = fit_calibration(pairs, (args.width, args.height), size)
        profile["camera_controls"] = {"fps": args.fps,
                                      "exposure_us": args.exposure_us,
                                      "gain": args.gain}
        save_calibration(profile, args.calibration)
        print(f"Saved {args.calibration}: {len(pairs)}/9 points, mean error {profile['mean_error_px']:.1f}px, max {profile['max_error_px']:.1f}px")
    finally:
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()


def run(args):
    size = display_size(args.screen)
    H, profile = load_calibration(args.calibration, (args.width, args.height), size)
    current_controls = {"fps": args.fps, "exposure_us": args.exposure_us,
                        "gain": args.gain}
    if profile.get("camera_controls") != current_controls:
        raise RuntimeError("Camera FPS, exposure, or gain differs from calibration; use the same settings or recalibrate.")
    if args.output == "touch" and not Path("/dev/uinput").exists():
        raise RuntimeError("/dev/uinput is absent; load the uinput kernel module first.")
    print("WARNING: Without an IR-pass filter, moving visible content can produce false touches.")
    camera = camera_open(*camera_settings(args))
    output = None
    tracker = PulseTracker(args.release_ms)
    background = None
    blank = np.zeros((size[1], size[0], 3), np.uint8) if args.output == "canvas" else None
    try:
        if args.output == "touch":
            output = TouchOutput(*size)
        if args.preview:
            cv_window("SMARTIR camera preview")
        if blank is not None:
            cv_window("SMARTIR whiteboard", True)
            cv2.imshow("SMARTIR whiteboard", blank)
            cv2.waitKey(250)  # establish the dark canvas before sampling background
        # Start with a pen-off reference. A held remote will be absorbed into
        # this baseline and will not produce a touch until re-pressed.
        frames = [gray(camera.capture_array("main")) for _ in range(12)]
        background = np.median(frames, axis=0).astype(np.uint8)
        previous = None
        last_log = 0
        while True:
            rgb = camera.capture_array("main")
            current = gray(rgb)
            spot, mask, reason = find_spot(current, background, min_delta=args.min_delta)
            point = map_point(H, spot, size) if spot is not None else None
            event, smoothed = tracker.update(point, time.monotonic())
            if output:
                output.emit(event, smoothed)
            if blank is not None:
                if event == "down" and smoothed is not None:
                    p = tuple(round(v) for v in smoothed)
                    cv2.circle(blank, p, 4, (255, 255, 255), -1)
                    previous = p
                elif event == "move" and smoothed is not None and previous is not None:
                    p = tuple(round(v) for v in smoothed)
                    cv2.line(blank, previous, p, (255, 255, 255), 4, cv2.LINE_AA)
                    previous = p
                elif event == "up":
                    previous = None
                cv2.imshow("SMARTIR whiteboard", blank)
            if args.preview:
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if spot is not None:
                    cv2.circle(preview, tuple(round(v) for v in spot), 9, (0, 255, 0), 2)
                cv2.putText(preview, f"{reason} | {event}", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
                cv2.imshow("SMARTIR camera preview", preview)
            if not tracker.active:
                # update only background regions not currently changing
                stable = mask == 0
                background[stable] = np.uint8(.97*background[stable] + .03*current[stable])
            if args.preview or blank is not None:
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if key == ord('c') and blank is not None:
                    blank[:] = 0
            elif time.monotonic()-last_log > 2:
                print(f"tracking: {reason}, {event}", flush=True)
                last_log = time.monotonic()
    finally:
        if output:
            output.close()
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()


def snapshot(args):
    camera = camera_open(*camera_settings(args))
    try:
        rgb = camera.capture_array("main")
        destination = Path(args.snapshot).resolve()
        if not cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {destination}")
        print(f"Camera image saved: {destination}")
    finally:
        camera.stop()
        camera.close()


def probe(args):
    """Console-only hardware check suitable for Raspberry Pi OS Lite."""
    camera = camera_open(*camera_settings(args))
    try:
        print("Keep the remote OFF and the screen steady for 2 seconds...", flush=True)
        frames = []
        started = time.monotonic()
        while time.monotonic()-started < 2:
            frames.append(gray(camera.capture_array("main")))
        background = np.median(frames, axis=0).astype(np.uint8)
        print("Now aim the remote LED at the camera, near the HDMI display, and hold a button for 10 seconds.", flush=True)
        hits, peaks, points = 0, [], []
        started = time.monotonic()
        while time.monotonic()-started < 10:
            current = gray(camera.capture_array("main"))
            peak = int(cv2.subtract(current, background).max())
            peaks.append(peak)
            spot, _, _ = find_spot(current, background, min_delta=args.min_delta)
            if spot is not None:
                hits += 1
                points.append(spot)
        print(f"Detected {hits} spot frames of {len(peaks)}; maximum brightness rise {max(peaks)} / 255.")
        if points:
            print(f"Median camera position: {tuple(np.median(points, axis=0).round(1))}")
        else:
            print("No usable pulses. Adjust exposure/gain, lens focus, distance, or remote aim before calibrating.")
    finally:
        camera.stop()
        camera.close()


def doctor(args):
    print(f"Platform: {platform.platform()}")
    print(f"Desktop: DISPLAY={os.environ.get('DISPLAY', '')!r}; WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')!r}")
    print(f"/dev/uinput: {'present' if Path('/dev/uinput').exists() else 'missing'}")
    for name in ("picamera2", "evdev", "tkinter"):
        try:
            __import__(name)
            print(f"{name}: installed")
        except ImportError:
            print(f"{name}: MISSING")
    for cmd in ("rpicam-hello", "libcamera-hello"):
        if shutil_which(cmd):
            result = subprocess.run([cmd, "--list-cameras"], capture_output=True, text=True, timeout=15)
            print(f"{cmd} --list-cameras:\n{result.stdout or result.stderr}")
            break
    if args.camera_test:
        camera = camera_open(*camera_settings(args))
        try:
            frame = camera.capture_array("main")
            print(f"Camera capture: OK {frame.shape}; grayscale range {gray(frame).min()}..{gray(frame).max()}")
            metadata = camera.capture_metadata()
            print("Actual camera controls:",
                  {key: metadata.get(key) for key in ("ExposureTime", "AnalogueGain", "FrameDuration")})
        finally:
            camera.stop()
            camera.close()


def shutil_which(command):
    from shutil import which
    return which(command)


def gui(args):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("SMARTIR control")
    root.geometry("440x300")
    tk.Label(root, text="SMARTIR • OV5647 whiteboard", font=("Sans", 17, "bold")).pack(pady=12)
    tk.Label(root, text="Release remote before each calibration target.\nPress its LED against the displayed crosshair.").pack(pady=4)
    process = None
    def launch(command):
        nonlocal process
        if process is not None and process.poll() is None:
            messagebox.showwarning("SMARTIR", "Stop the running mode first.")
            return
        command_args = [sys.executable, str(Path(__file__).resolve()), command,
                        "--calibration", str(args.calibration), "--width", str(args.width),
                        "--height", str(args.height), "--fps", str(args.fps),
                        "--exposure-us", str(args.exposure_us), "--gain", str(args.gain),
                        "--min-delta", str(args.min_delta)]
        if command == "run":
            command_args += ["--output", "canvas", "--preview"]
        if args.screen:
            command_args += ["--screen", args.screen]
        process = subprocess.Popen(command_args)
    tk.Button(root, text="1  Calibrate screen", command=lambda: launch("calibrate"), width=28).pack(pady=5)
    tk.Button(root, text="2  Test whiteboard + preview", command=lambda: launch("run"), width=28).pack(pady=5)
    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
    tk.Button(root, text="Stop", command=stop, width=28).pack(pady=5)
    tk.Label(root, text="For desktop-wide touch use CLI: run --output touch\nNo IR filter: use only on a static/dark display.", fg="#8a4b00").pack(pady=9)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop(), root.destroy()))
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "snapshot", "probe", "calibrate", "run", "gui"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure-us", type=int, default=15000)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--min-delta", type=int, default=28)
    parser.add_argument("--release-ms", type=int, default=170)
    parser.add_argument("--target-timeout", type=float, default=8.0)
    parser.add_argument("--screen", help="Override display resolution WIDTHxHEIGHT")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--snapshot", default="smartir-snapshot.png")
    parser.add_argument("--camera-test", action="store_true")
    parser.add_argument("--output", choices=("canvas", "touch"), default="canvas")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    if min(args.width, args.height) < 160 or args.fps < 1 or args.exposure_us < 100 or args.gain <= 0 or args.min_delta < 1:
        parser.error("Invalid camera/detection settings")
    try:
        if args.command == "doctor": doctor(args)
        elif args.command == "snapshot": snapshot(args)
        elif args.command == "probe": probe(args)
        elif args.command == "calibrate": calibrate(args)
        elif args.command == "run": run(args)
        elif args.command == "gui": gui(args)
    except KeyboardInterrupt:
        print("Stopped")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"SMARTIR error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
