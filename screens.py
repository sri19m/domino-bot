"""Learn and recognise non-gameplay screens (home, start dialog, win, popups)."""
import json, os, cv2
import vision
DIR = os.path.join(os.path.dirname(__file__), "screens")
INDEX = os.path.join(DIR, "screens.json")

def _load():
    """Learned screens. A missing folder, missing/empty/invalid screens.json all mean 'no screens'."""
    try:
        with open(INDEX, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    keys = ("name", "file", "roi", "tap")
    return [s for s in data if isinstance(s, dict) and all(k in s for k in keys)]

def match(img):
    H, W = img.shape[:2]
    for s in _load():
        tpl = cv2.imread(os.path.join(DIR, s["file"]))
        if tpl is None:
            continue
        x, y, w, h = s["roi"]
        x0, y0 = int(x * W), int(y * H)
        patch = img[y0:y0 + int(h * H), x0:x0 + int(w * W)]
        if patch.shape[:2] != tpl.shape[:2]:
            patch = cv2.resize(patch, (tpl.shape[1], tpl.shape[0]))
        score = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED).max()
        if score > 0.8:
            return s
    return None

def teach(img):
    """Ask the user to draw a box around something unique on the screen, then click where to tap."""
    if vision.find_hand(vision.downscale(vision.normalize(img)[0])[0]) is not None:
        print("Refusing to teach: a hand tile is visible, so this is a gameplay screen. "
              "Only teach menus, level start, win or popups.")
        return None
    os.makedirs(DIR, exist_ok=True)
    H, W = img.shape[:2]
    scale = 1100 / W
    view = cv2.resize(img, None, fx=scale, fy=scale)
    print("\nUNKNOWN SCREEN. Do NOT teach a gameplay screen (tiles on the board, even mid-animation).")
    print("To skip: press Esc or c in the selector, or leave the name empty.")
    print("1) Drag a box around a fixed, unique element (title text/button), press ENTER.")
    x, y, w, h = cv2.selectROI("teach: select unique region", view, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        return None
    clicked = []
    def cb(ev, mx, my, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicked.append((mx, my))
    cv2.namedWindow("teach: click where to tap")
    cv2.setMouseCallback("teach: click where to tap", cb)
    print("2) Click the spot the bot should tap on this screen (close the window with any key to cancel).")
    while not clicked:
        cv2.imshow("teach: click where to tap", view)
        if cv2.waitKey(30) != -1:
            break
    cv2.destroyAllWindows()
    if not clicked:
        return None
    name = input("Name for this screen (e.g. level_start, win, out_of_tiles; empty = skip): ").strip()
    if not name:
        print("Skipped, nothing saved.")
        return None
    if any(c in name for c in '/\\:'):
        print("Invalid name, nothing saved.")
        return None
    rx, ry, rw, rh = x / view.shape[1], y / view.shape[0], w / view.shape[1], h / view.shape[0]
    x0, y0 = int(rx * W), int(ry * H)
    patch = img[y0:y0 + int(rh * H), x0:x0 + int(rw * W)]
    fname = f"{name}.png"
    cv2.imwrite(os.path.join(DIR, fname), patch)
    entry = {"name": name, "file": fname, "roi": [rx, ry, rw, rh],
             "tap": [clicked[0][0] / view.shape[1], clicked[0][1] / view.shape[0]]}
    data = [s for s in _load() if s["name"] != name] + [entry]
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved '{name}'.")
    return entry
