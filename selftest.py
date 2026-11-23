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

def main():
    base = scene()
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/synthetic.png", base)
    ok = check(base, "1024x477")
    ok &= check(cv2.resize(base, (2316, 1080), interpolation=cv2.INTER_CUBIC), "2316x1080")
    ok &= check(cv2.resize(base, (3088, 1440), interpolation=cv2.INTER_CUBIC), "3088x1440")
    ok &= check(cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE), "portrait (rotated 90 cw)")
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
