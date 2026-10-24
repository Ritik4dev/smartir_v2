# SMARTIR laptop/webcam test

This folder is the PC version. It reuses the detection, homography, and
smoothing functions from `../smartir_v2.py`, but opens an ordinary webcam and
saves its **own** `pc-calibration.json`. Do not copy calibration between the
Pi and PC: the camera and geometry differ.
It also accepts an IP camera's live HTTP(S) MJPEG or RTSP video stream. An IP
camera must be on a reachable network; a webpage or single JPEG snapshot URL
is not a continuous video stream.
The PC detector uses the already-required OpenCV connected-components API
to find small, newly brighter spots, including one-pixel LEDs. No separate
heavy tracking package or model download is needed.

On Windows, from this folder:

```powershell
py -m pip install -r requirements.txt
py smartir_pc.py
```

Or if `python` is your working command, replace `py` with `python`.
The GUI buttons guide you through preview, calibration, canvas test, and
optional desktop mouse control. Start with camera preview. If the external
USB webcam is index 1, change the index in the GUI or use `--camera 1`.
To use an IP camera, paste its **video stream URL** into the optional GUI field.
That overrides the webcam index. A typical phone IP-camera app might expose
`http://192.168.1.5:8080/video`; use the actual URL shown by your app. A
camera may instead expose a `rtsp://...` URL. Keep the phone and PC on the
same reachable network. The GUI passes the URL to its child via an environment
variable; the calibration file stores only its SHA-256 fingerprint, not the
URL or any embedded password.

CLI equivalents:

```powershell
py smartir_pc.py doctor --camera 0
py smartir_pc.py preview --camera 0
py smartir_pc.py calibrate --camera 0
py smartir_pc.py run --camera 0 --output canvas
py smartir_pc.py run --camera 0 --output mouse
py smartir_pc.py doctor --stream-url "http://192.168.1.5:8080/video"
py smartir_pc.py preview --stream-url "rtsp://192.168.1.5/live"
```

For a CLI URL containing a password, set `SMARTIR_STREAM_URL` in your shell
instead of writing it in a command (where it may appear in process listings).
The IP camera's stream resolution is controlled by the camera app, not the
`--width` and `--height` webcam options. If the stream URL or resolution
changes, recalibrate. OpenCV's FFmpeg backend is used where available, with
a generic backend fallback. No additional Python dependency is required.

The camera **must see the whole laptop display** for calibration. A built-in
webcam mounted above a laptop screen generally faces the user and cannot see
its own screen plane; place an external USB webcam facing the display. During
calibration, place the remote's emitting LED at each crosshair and aim it
toward that webcam. Four corner targets are mandatory; five more improve and
validate the mapping. Release the remote before each new target appears.
The calibration screen now includes a live camera view. A green circle marks
the detected light in that view; the ring around the target fills as the
camera repeatedly sees a stable spot. Wait for **POINT CALIBRATED** before
moving to the next target. The test whiteboard has the same live annotated
camera view in a corner; the inset itself is not saved into the drawing.
After a successful calibration the app automatically opens the blank test
whiteboard; press **C** to clear it or **Q** to close it. CLI users can pass
`--no-auto-test` to stop after saving the calibration instead.
The detector now rejects elongated bright streaks and prefers a compact spot
near the one it was already following, so a brighter lamp elsewhere is less
likely to steal the track. A green ring is a fresh detection; an amber ring
marked **SIGNAL HELD** means the camera missed a frame but tracking is being
held briefly for a pulsing TV remote. This does not prove the held light is
still on.
If the LED is visible in the preview but has no green circle, lower the GUI
brightness threshold from 18 toward 10, then preview again. Lower values are
more sensitive but may also catch moving visible reflections. Keep the remote
button released while each new calibration target gathers its background.

This is a bright-spot detector, not a spectral IR detector. If your webcam
can see the remote LED, testing is possible, but an unfiltered webcam can
also mistake changing visible light for the pen. Use the contained canvas
first. Only enable desktop mouse output after verifying calibration and
keeping the display content steady. Press Q or Esc in the OpenCV window to
stop; the mouse button is released on normal shutdown.

Webcam exposure units differ by model. `--exposure` and `--gain` are optional
backend-specific controls; if you change either, recalibrate with matching
settings. The app reports the actual frame size and rejects a calibration
made for a different size or screen resolution.
