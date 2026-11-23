import subprocess, numpy as np, cv2

class Device:
    def __init__(self, serial=None):
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.w = self.h = None

    def screenshot(self):
        raw = subprocess.run(self.base + ["exec-out", "screencap", "-p"],
                             capture_output=True, check=True).stdout
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Screenshot failed. Is the phone connected and USB debugging authorised?")
        self.h, self.w = img.shape[:2]
        return img

    def tap_px(self, x, y):
        subprocess.run(self.base + ["shell", "input", "tap", str(int(x)), str(int(y))], check=True)

    def tap_frac(self, fx, fy):
        if self.w is None:
            self.screenshot()
        self.tap_px(fx * self.w, fy * self.h)
