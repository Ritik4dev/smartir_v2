"""Launcher tests; no Pi hardware or package installation required."""

import io
import unittest
from unittest.mock import patch

import smartir


class LauncherTests(unittest.TestCase):
    def test_package_manifest_is_safe(self):
        self.assertEqual(smartir.apt_packages(), [
            "python3-picamera2", "python3-opencv", "python3-numpy",
            "python3-tk", "python3-evdev",
        ])

    def test_no_install_when_imports_work(self):
        with patch.object(smartir, "on_raspberry_pi", return_value=True), \
             patch.object(smartir, "missing_imports", return_value=None), \
             patch.object(smartir.subprocess, "run") as run:
            smartir.ensure_dependencies("/usr/bin/python3")
            run.assert_not_called()

    def test_noninteractive_missing_packages_does_not_install(self):
        with patch.object(smartir, "on_raspberry_pi", return_value=True), \
             patch.object(smartir, "missing_imports", return_value="missing cv2"), \
             patch.object(smartir.sys.stdin, "isatty", return_value=False), \
             patch.object(smartir.subprocess, "run") as run, \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "Run interactively"):
                smartir.ensure_dependencies("/usr/bin/python3")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
