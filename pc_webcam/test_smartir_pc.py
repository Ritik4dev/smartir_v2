"""Hardware-free tests for the PC webcam application."""

import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from smartir_pc import (CalibrationHold, WindowsMouse, annotated_camera,
                        calibration_screen, camera_controls, camera_inset,
                        gray_bgr, learn, load_led_profile, main, open_camera, open_ip_stream,
                        validate_stream_url)


class FakeUser32:
    def __init__(self):
        self.positions = []
        self.buttons = 0

    def SetCursorPos(self, x, y):
        self.positions.append((x, y))
        return 1

    def SendInput(self, count, pointer, size):
        self.buttons += count
        return count


class PcTests(unittest.TestCase):
    def test_teach_workflow_collects_confirmed_frames(self):
        off = np.full((90, 130, 3), 20, np.uint8)
        on = off.copy()
        cv2.circle(on, (50, 40), 2, (170, 20, 170), -1)
        args = SimpleNamespace(led_profile=Path("unused-profile.json"),
                               stream_url="", camera=0, width=130,
                               height=90, fps=30, exposure=None, gain=None,
                               min_delta=18, max_area=250)
        cap = unittest.mock.MagicMock()
        reads = [0]
        callback = [None]
        keys = [0]

        def next_frame(_cap):
            reads[0] += 1
            return off if reads[0] <= 17 else on

        def set_callback(_name, handler):
            callback[0] = handler

        def next_key(_delay):
            keys[0] += 1
            if keys[0] == 1:
                return 32
            if keys[0] == 18:
                callback[0](cv2.EVENT_LBUTTONDOWN, 50, 40, 0, None)
            return 13 if keys[0] == 80 else 0

        with patch("smartir_pc.open_camera", return_value=(cap, off, 700)), \
             patch("smartir_pc.read_frame", side_effect=next_frame), \
             patch("smartir_pc.view"), \
             patch("smartir_pc.cv2.resizeWindow"), \
             patch("smartir_pc.cv2.setMouseCallback", side_effect=set_callback), \
             patch("smartir_pc.cv2.imshow"), \
             patch("smartir_pc.cv2.waitKey", side_effect=next_key), \
             patch("smartir_pc.cv2.destroyAllWindows"), \
             patch("smartir_pc.save_calibration") as saved:
            self.assertTrue(learn(args))
        profile = saved.call_args.args[0]
        self.assertGreaterEqual(profile["sample_count"], 8)
        self.assertEqual(profile["camera_size"], [130, 90])
        cap.release.assert_called_once()

    def test_learned_profile_is_bound_to_camera(self):
        args = SimpleNamespace(led_profile=Path("sample-led.json"),
                               stream_url="", camera=0, width=640,
                               height=480, fps=30, exposure=None, gain=None)
        profile = {"version": 1, "camera_size": [640, 480],
                   "camera_controls": camera_controls(args),
                   "bgr_chroma": [.4, .2, .4],
                   "chroma_tolerance": .3, "color_reliable": True}
        with patch("smartir_pc.Path.read_text", return_value=json.dumps(profile)):
            self.assertTrue(load_led_profile(args, (640, 480))["color_reliable"])
            args.camera = 1
            with self.assertRaisesRegex(RuntimeError, "Camera source"):
                load_led_profile(args, (640, 480))

    def test_calibration_opens_canvas_after_success(self):
        with patch("smartir_pc.calibrate") as calibrate, patch("smartir_pc.run") as run:
            self.assertEqual(main(["calibrate"]), 0)
        calibrate.assert_called_once()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0].output, "canvas")

    def test_calibration_can_stop_without_auto_test(self):
        with patch("smartir_pc.calibrate") as calibrate, patch("smartir_pc.run") as run:
            self.assertEqual(main(["calibrate", "--no-auto-test"]), 0)
        calibrate.assert_called_once()
        run.assert_not_called()

    def test_ip_stream_url_validation(self):
        for url in ("http://192.168.1.5:8080/video", "https://cam.example/live",
                    "rtsp://user:pass@192.168.1.6/stream"):
            self.assertEqual(validate_stream_url(url), url)
        for url in ("", "file:///tmp/video.mp4", "C:\\video.mp4", "http:///video"):
            with self.assertRaises(ValueError):
                validate_stream_url(url)

    def test_ip_stream_opens_and_reads_frame(self):
        frame = np.zeros((48, 64, 3), np.uint8)
        fake = unittest.mock.MagicMock()
        fake.isOpened.return_value = True
        fake.read.return_value = (True, frame)
        fake.get.return_value = 1900
        with patch("smartir_pc.cv2.VideoCapture", return_value=fake) as capture:
            cap, result, backend = open_ip_stream("http://camera.local/video")
        self.assertIs(cap, fake)
        self.assertIs(result, frame)
        self.assertEqual(backend, 1900)
        self.assertEqual(capture.call_args.args[1], cv2.CAP_FFMPEG)

    def test_ip_stream_failure_releases_capture(self):
        fake = unittest.mock.MagicMock()
        fake.isOpened.return_value = True
        fake.read.return_value = (False, None)
        with patch("smartir_pc.cv2.VideoCapture", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "no video frames"):
                open_ip_stream("rtsp://camera.local/live")
        fake.release.assert_called_once()

    def test_ip_camera_profile_does_not_store_password(self):
        args = SimpleNamespace(stream_url="rtsp://alice:secret@camera.local/live",
                               camera=0, width=640, height=480, fps=30,
                               exposure=None, gain=None, stream_timeout_ms=5000)
        controls = camera_controls(args)
        self.assertNotIn("secret", str(controls))
        with patch("smartir_pc.open_ip_stream", return_value=("cap", "frame", 1900)) as opening:
            self.assertEqual(open_camera(args)[0], "cap")
        opening.assert_called_once_with(args.stream_url, 5000)

    def test_calibration_hold_requires_stable_repeated_hits(self):
        hold = CalibrationHold(required=3)
        self.assertFalse(hold.update((20, 20), 0.0))
        self.assertFalse(hold.update(None, 0.05))
        self.assertFalse(hold.update((21, 20), 0.1))
        self.assertTrue(hold.update((20, 21), 0.2))
        self.assertEqual(hold.progress, 1)
        self.assertAlmostEqual(hold.point[0], 20)

    def test_calibration_hold_resets_on_jump_or_long_gap(self):
        hold = CalibrationHold(required=3)
        hold.update((20, 20), 0)
        hold.update((21, 20), .1)
        self.assertFalse(hold.update((100, 100), .2))
        self.assertEqual(len(hold.samples), 1)
        self.assertFalse(hold.update((101, 100), 1.0))
        self.assertEqual(len(hold.samples), 1)

    def test_camera_annotation_and_calibration_inset(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        self.assertEqual(annotated_camera(frame, (60, 50), "spot").shape, frame.shape)
        canvas = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(camera_inset(canvas.copy(), frame, (60, 50), "spot").shape,
                         canvas.shape)
        hold = CalibrationHold()
        hold.update((60, 50), 0)
        screen = calibration_screen((1280, 720), (120, 72), 0, frame,
                                    (60, 50), "spot", hold, "capture")
        self.assertEqual(screen.shape, canvas.shape)
        self.assertGreater(np.count_nonzero(screen), 0)

    def test_bgr_grayscale(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[:, :, 2] = 255
        self.assertTrue(np.array_equal(gray_bgr(frame),
                                       cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    def test_mouse_down_move_release(self):
        api = FakeUser32()
        mouse = WindowsMouse(user32=api)
        mouse.emit("down", (10.4, 20.6))
        mouse.emit("move", (12.2, 23.8))
        mouse.close()
        self.assertEqual(api.positions, [(10, 21), (12, 24)])
        self.assertEqual(api.buttons, 2)
        self.assertFalse(mouse.down)


if __name__ == "__main__":
    unittest.main()
