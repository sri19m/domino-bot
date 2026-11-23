import cv2, numpy as np
import config

WORK_W = 1024  # all detection runs on a downscaled copy

def downscale(img):
    s = WORK_W / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

def _crop(img, region):
    H, W = img.shape[:2]
    x, y, w, h = region
    x0, y0 = int(x * W), int(y * H)
    return img[y0:y0 + int(h * H), x0:x0 + int(w * W)], (x0, y0)

def _tile_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, config.TILE_V_MIN), (180, config.TILE_S_MAX, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # fill pips so tile is one blob
    return m

def _count_pips(half):
    hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)
    pip = ((hsv[..., 1] > 70) | (hsv[..., 2] < 130)).astype(np.uint8) * 255
    pip = cv2.morphologyEx(pip, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(pip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    min_a = 0.012 * half.shape[0] * half.shape[1]
    areas = [a for a in areas if a >= min_a]
    if not areas:
        return 0
    med = np.median(areas)
    # merged blobs count as several pips
    return min(12, int(sum(max(1, round(a / med)) for a in areas)))

def read_tile(img, contour):
    """Return (value_a, value_b, centre_xy) for one face-up tile contour."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), ang = rect
    long_, short_ = max(w, h), min(w, h)
    box = cv2.boxPoints(rect).astype(np.float32)
    # order box so that the first edge is the long side
    if np.linalg.norm(box[0] - box[1]) < np.linalg.norm(box[1] - box[2]):
        box = np.roll(box, 1, axis=0)
    L, S = int(long_), int(short_)
    if L < 10 or S < 10:
        return None
    dst = np.array([[0, 0], [L, 0], [L, S], [0, S]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warp = cv2.warpPerspective(img, M, (L, S))
    mx, my = int(L * 0.05), int(S * 0.12)
    warp = warp[my:S - my, mx:L - mx]
    mid = warp.shape[1] // 2
    a = _count_pips(warp[:, :mid - 2])
    b = _count_pips(warp[:, mid + 2:])
    return a, b, (cx, cy)

def find_hand(img_small):
    crop, (ox, oy) = _crop(img_small, config.HAND_REGION)
    cnts, _ = cv2.findContours(_tile_mask(crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1200:
        return None
    t = read_tile(crop, c)
    if t is None:
        return None
    a, b, (cx, cy) = t
    return {"vals": (a, b), "pos": (cx + ox, cy + oy)}

def find_board(img_small):
    crop, (ox, oy) = _crop(img_small, config.BOARD_REGION)
    cnts, _ = cv2.findContours(_tile_mask(crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500 or area > 25000:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        ratio = max(w, h) / max(1.0, min(w, h))
        fill = area / max(1.0, w * h)
        if not (1.5 < ratio < 2.6) or fill < 0.75:
            continue
        t = read_tile(crop, c)
        if t:
            a, b, (tx, ty) = t
            tiles.append({"vals": (a, b), "pos": (tx + ox, ty + oy)})
    return tiles

def annotate(img_small, hand, tiles, path):
    out = img_small.copy()
    for t in tiles:
        cv2.putText(out, f"{t['vals'][0]}|{t['vals'][1]}", (int(t["pos"][0]) - 20, int(t["pos"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if hand:
        cv2.putText(out, f"H {hand['vals'][0]}|{hand['vals'][1]}", (int(hand["pos"][0]) - 30, int(hand["pos"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(path, out)
