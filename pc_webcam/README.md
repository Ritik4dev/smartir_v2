# SMARTIR laptop/webcam test

This folder is the PC version. It reuses the detection, homography, and
smoothing functions from `../smartir_v2.py`, but opens an ordinary webcam and
saves its **own** `pc-calibration.json`. Do not copy calibration between the
Pi and PC: the camera and geometry differ.

On Windows, from this folder:

```powershell
py -m pip install -r requirements.txt
py smartir_pc.py
```

Or if `python` is your working command, replace `py` with `python`.
The GUI buttons guide you through preview, calibration, canvas test, and
optional desktop mouse control. Start with camera preview. If the external
USB webcam is index 1, change the index in the GUI or use `--camera 1`.

CLI equivalents:

```powershell
py smartir_pc.py doctor --camera 0
py smartir_pc.py preview --camera 0
py smartir_pc.py calibrate --camera 0
py smartir_pc.py run --camera 0 --output canvas
py smartir_pc.py run --camera 0 --output mouse
```

The camera **must see the whole laptop display** for calibration. A built-in
webcam mounted above a laptop screen generally faces the user and cannot see
its own screen plane; place an external USB webcam facing the display. During
calibration, place the remote's emitting LED at each crosshair and aim it
toward that webcam. Four corner targets are mandatory; five more improve and
validate the mapping. Release the remote before each new target appears.

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
