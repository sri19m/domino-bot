"""Synthetic detection test: python selftest.py  (no phone needed).

Draws dominoes on a green background in the 1024x477 reference layout, including 45 degree tiles,
a vertical hand tile, a face-down deck and an undo button, then checks the values vision reads.
Also runs the same scene at phone resolutions and rotated to portrait. Any PNG/JPG in ./reference
is run through the pipeline too (smoke test, prints results, only fails on exceptions).
"""
import glob, os, sys
import cv2, numpy as np
import vision

PIPS = {
    0: [], 1: [(.5, .5)], 2: [(.25, .25), (.75, .75)], 3: [(.2, .2), (.5, .5), (.8, .8)],
    4: [(.25, .25), (.75, .25), (.25, .75), (.75, .75)],
    5: [(.25, .25), (.75, .25), (.5, .5), (.25, .75), (.75, .75)],
    6: [(.25, .2), (.75, .2), (.25, .5), (.75, .5), (.25, .8), (.75, .8)],
}
COLORS = [(230, 120, 40), (200, 60, 150), (40, 140, 250), (180, 80, 240), (240, 200, 90)]  # BGR: blue, purple, orange, pink, light blue

def tile_patch(a, b, L, S):
    """Upright (horizontal) face-up tile on a BGR patch plus its mask."""
    patch = np.zeros((S, L, 3), np.uint8)
    mask = np.zeros((S, L), np.uint8)
    cv2.rectangle(mask, (0, 0), (L - 1, S - 1), 255, -1)
    patch[:] = (238, 245, 250)
    cv2.line(patch, (L // 2, int(S * .12)), (L // 2, int(S * .88)), (170, 170, 170), 2)
    r = max(2, int(S * 0.075))
    for half, v in enumerate((a, b)):
        for i, (px, py) in enumerate(PIPS[v]):
            x = int((half * .5 + px * .5) * L)
            y = int((0.1 + py * .8) * S)
            cv2.circle(patch, (x, y), r, COLORS[(v + i) % len(COLORS)], -1, cv2.LINE_AA)
    return patch, mask

def paste(canvas, a, b, cx, cy, L, S, angle):
    patch, mask = tile_patch(a, b, L, S)
    M = cv2.getRotationMatrix2D((L / 2, S / 2), angle, 1.0)
    M[:, 2] += (cx - L / 2, cy - S / 2)
    H, W = canvas.shape[:2]
    p = cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST) > 0
    canvas[m] = p[m]

def face_down(canvas, x0, y0, w, h):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (235, 240, 245), -1)
    cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + w - 4, y0 + h - 4), (150, 40, 150), -1)

BOARD = [  # (a, b, cx, cy, angle) in 1024x477 reference pixels
    (3, 5, 330, 100, 0), (6, 2, 480, 100, 0), (1, 4, 640, 110, 0), (0, 6, 330, 200, 0),
    (5, 5, 600, 330, 0), (2, 3, 720, 220, 45), (4, 1, 290, 330, -45), (6, 3, 480, 230, 90),
]
HAND = (4, 2)

def scene():
    img = np.zeros((477, 1024, 3), np.uint8)
    img[:] = (60, 140, 60)
    for a, b, cx, cy, ang in BOARD:
        paste(img, a, b, cx, cy, 110, 58, ang)
    for i in range(3):  # deck of face-down tiles
        face_down(img, 440 + i * 3, 400 + i * 3, 60, 42)
    paste(img, HAND[0], HAND[1], 655, 425, 57, 40, 90)  # vertical hand tile
    cv2.circle(img, (790, 425), 22, (120, 120, 120), -1)  # undo button
    return img

def norm(vals):
    return tuple(sorted(vals))

def check(img, label):
    img, _ = vision.normalize(img)
    small, _ = vision.downscale(img)
    hand = vision.find_hand(small)
    tiles = vision.find_board(small)
    got_h = hand and norm(hand["vals"])
    got_b = sorted(norm(t["vals"]) for t in tiles)
    want_b = sorted(norm((a, b)) for a, b, *_ in BOARD)
    ok = got_h == norm(HAND) and got_b == want_b
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: hand={got_h} board={got_b}")
    if not ok:
        print(f"        wanted hand={norm(HAND)} board={want_b}")
    return ok

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_FIXTURE = os.path.join(HERE, "tests", "fixtures", "real_level_1.png")
# Ground truth for the real 2316x1080 capture: values and tap centres in full-res pixels.
REAL_HAND = ((1, 3), (1493, 957))
REAL_BOARD = [
    ((2, 3), (960, 390), True),    # partly covered by the tutorial hand: off-by-one tolerated with a warning
    ((1, 6), (570, 630), False),
    ((5, 6), (1160, 775), False),
    ((0, 5), (1355, 395), False),
    ((0, 4), (1745, 625), False),
]
POS_TOL = 40

def check_real(path=REAL_FIXTURE):
    """Regression test on a real phone capture. Skipped (passes) when the fixture is missing."""
    img = cv2.imread(path)
    if img is None:
        print(f"[SKIP] real capture: {path} not found")
        return True
    im, _ = vision.normalize(img)
    small, s = vision.downscale(im)
    hand = vision.find_hand(small)
    tiles = [(norm(t["vals"]), (t["pos"][0] / s, t["pos"][1] / s)) for t in vision.find_board(small)]
    errs, warns = [], []
    near = lambda p, q: abs(p[0] - q[0]) <= POS_TOL and abs(p[1] - q[1]) <= POS_TOL
    if not hand:
        errs.append("no hand")
    else:
        hp = (hand["pos"][0] / s, hand["pos"][1] / s)
        if norm(hand["vals"]) != REAL_HAND[0] or not near(hp, REAL_HAND[1]):
            errs.append(f"hand {norm(hand['vals'])} at {hp[0]:.0f},{hp[1]:.0f}")
    used = set()
    for want, pos, lenient in REAL_BOARD:
        idx = [i for i, (_, p) in enumerate(tiles) if i not in used and near(p, pos)]
        if not idx:
            errs.append(f"no tile near {pos} (want {want})")
            continue
        i = idx[0]
        used.add(i)
        got = tiles[i][0]
        if got == want:
            continue
        diff = sum(abs(a - b) for a, b in zip(got, want))
        if lenient and diff == 1:
            warns.append(f"tile near {pos} read {got}, expected {want} (occluded)")
        else:
            errs.append(f"tile near {pos} read {got}, expected {want}")
    extra = [tiles[i] for i in range(len(tiles)) if i not in used]
    if extra:
        errs.append(f"unexpected tiles {[(v, (int(p[0]), int(p[1]))) for v, p in extra]}")
    for w in warns:
        print(f"        WARNING: {w}")
    ok = not errs
    print(f"[{'PASS' if ok else 'FAIL'}] real capture {os.path.basename(path)}: "
          f"hand={hand and norm(hand['vals'])} board={sorted(v for v, _ in tiles)}")
    for e in errs:
        print(f"        {e}")
    return ok

def check_screens():
    """screens._load must treat a missing folder, missing, empty or invalid screens.json as no screens."""
    import tempfile
    import screens
    old_dir, old_index = screens.DIR, screens.INDEX
    ok = True
    try:
        with tempfile.TemporaryDirectory() as d:
            cases = {"missing folder": None, "missing screens.json": "", "empty file": b"",
                     "invalid json": b"{not json", "not a list": b'{"a": 1}'}
            for label, content in cases.items():
                sd = os.path.join(d, label.replace(" ", "_"))
                if content is not None:
                    os.makedirs(sd)
                    if content != "":
                        with open(os.path.join(sd, "screens.json"), "wb") as f:
                            f.write(content)
                screens.DIR, screens.INDEX = sd, os.path.join(sd, "screens.json")
                try:
                    got = screens._load()
                    hit = screens.match(np.zeros((100, 200, 3), np.uint8))
                    good = got == [] and hit is None
                except Exception as e:  # noqa: BLE001 - any crash is a failure here
                    good, got = False, repr(e)
                ok &= good
                if not good:
                    print(f"        screens {label}: got {got}")
    finally:
        screens.DIR, screens.INDEX = old_dir, old_index
    print(f"[{'PASS' if ok else 'FAIL'}] screens: missing/empty/invalid index treated as no screens")
    return ok

class FakeDev:
    """Serves a scripted list of frames (the last one repeats) and records taps."""
    def __init__(self, frames):
        self.frames, self.i, self.taps = frames, 0, []
    def screenshot(self):
        img = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return img.copy()
    def tap_px(self, x, y):
        self.taps.append(("px", x, y))
    def tap_frac(self, fx, fy):
        self.taps.append(("frac", fx, fy))

def check_loop():
    """Offline simulation of the main loop decisions with a fake device and fake clock."""
    import main
    real = cv2.imread(REAL_FIXTURE)
    if real is None:
        print(f"[SKIP] loop: {REAL_FIXTURE} not found")
        return True
    blank = np.full(real.shape, 30, np.uint8)
    menu = np.full(real.shape, 90, np.uint8)
    cv2.putText(menu, "PLAY", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 12)
    errs = []

    def run(frames, steps, known=None):
        dev, now, calls = FakeDev(frames), [0.0], {"match": 0, "teach": 0}
        def sleep(t): now[0] += t
        def match(img):
            calls["match"] += 1
            return known
        def teach(img):
            calls["teach"] += 1
            return None
        bot = main.Bot(dev, match=match, teach=teach, sleep=sleep, clock=lambda: now[0])
        res = [bot.step() for _ in range(steps)]
        return res, calls, dev, now[0]

    # 1. real gameplay: taps, never match/teach, even for many steps (stuck frames draw instead)
    moving = []
    for k in range(12):  # board changes between frames, as after real moves
        f = real.copy()
        f[:300, :200 + 100 * k] = 0
        moving.append(f)
    res, calls, dev, _ = run(moving, 12, known={"name": "x", "tap": [0.5, 0.5]})
    if calls["match"] or calls["teach"]:
        errs.append(f"gameplay frames reached match/teach {calls}")
    if not dev.taps or any(t[0] == "frac" for t in dev.taps):
        errs.append("gameplay produced no taps or tapped a learned-screen position")
    # 1b. frozen gameplay: draws first, only then is the screen treated as non-gameplay
    res, calls, dev, _ = run([real], 12, known=None)
    px = [x for x in dev.taps if x[0] == "px"]
    if not px or calls["match"] != 1:
        errs.append(f"frozen gameplay: {calls} taps={len(px)}")
    # 2. transient frames, then gameplay: no prompt, no match
    res, calls, dev, _ = run([blank, blank, blank, real], 1)
    if res != ["gameplay"] or calls["match"] or calls["teach"]:
        errs.append(f"transient frames: {res} {calls}")
    # 3. non-gameplay: polls ~POLL_TOTAL before match/teach, then prompts once per step
    res, calls, dev, t = run([menu], 1)
    if t < main.POLL_TOTAL or calls != {"match": 1, "teach": 1} or res != ["skipped"]:
        errs.append(f"non-gameplay: {res} {calls} t={t}")
    # 4. non-gameplay with a learned screen: taps it, no teach
    res, calls, dev, _ = run([menu], 1, known={"name": "play", "tap": [0.5, 0.5]})
    if res != ["known"] or calls["teach"] or dev.taps != [("frac", 0.5, 0.5)]:
        errs.append(f"known screen: {res} {calls} {dev.taps}")
    # 5. teach refuses a gameplay frame
    import screens
    if screens.teach(real) is not None:
        errs.append("teach accepted a gameplay frame")
    for e in errs:
        print("        " + e)
    print(f"[{'PASS' if not errs else 'FAIL'}] loop: gameplay never teaches/taps screens, transient frames poll first")
    return not errs

def main():
    base = scene()
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/synthetic.png", base)
    ok = check(base, "1024x477")
    ok &= check(cv2.resize(base, (2316, 1080), interpolation=cv2.INTER_CUBIC), "2316x1080")
    ok &= check(cv2.resize(base, (3088, 1440), interpolation=cv2.INTER_CUBIC), "3088x1440")
    ok &= check(cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE), "portrait (rotated 90 cw)")
    ok &= check_real()
    ok &= check_screens()
    ok &= check_loop()
    for p in sorted(glob.glob("reference/*.png") + glob.glob("reference/*.jpg")):
        img = cv2.imread(p)
        im, _ = vision.normalize(img)
        small, _ = vision.downscale(im)
        h = vision.find_hand(small)
        t = vision.find_board(small)
        print(f"[ref ] {p}: hand={h and h['vals']} board={[x['vals'] for x in t]}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
