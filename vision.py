import cv2, numpy as np
import config

WORK_W = 1024  # all detection runs on a downscaled copy

def downscale(img):
    s = WORK_W / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

def normalize(img):
    """Rotate portrait captures to landscape. Returns (img, rot) where rot is None or 'ccw'."""
    if img.shape[0] > img.shape[1]:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), "ccw"
    return img, None

def unrotate_point(x, y, rot, orig_w):
    """Map a pixel in the normalized frame back to the original capture (orig_w = original width)."""
    if rot == "ccw":
        # ccw rotation of a W-wide image: x' = y, y' = W - x  =>  x = W - y', y = x'
        return orig_w - y, x
    return x, y

# Tile geometry is expressed relative to WORK_W**2 so it scales with the working resolution.
# Reference (1024 wide): board tile ~110x58 = 6400 px, hand tile ~41x57 = 2300 px.
_REF = float(WORK_W * WORK_W)
BOARD_AREA = (0.0025 * _REF, 0.03 * _REF)
HAND_AREA = (0.0008 * _REF, 0.012 * _REF)
BOARD_RATIO = (1.4, 2.8)
HAND_RATIO = (1.2, 3.2)
MIN_SOLID = 0.7        # fraction of the blob that is cream; face-down tile borders fail this
MIN_FILL = 0.7         # blob area / min-area-rect area
# (S max, V min) tried in order; second is a looser fallback for bright or washed-out displays
MASK_TIERS = [(None, None), (100, 160)]

def _crop(img, region):
    H, W = img.shape[:2]
    x, y, w, h = region
    x0, y0 = int(x * W), int(y * H)
    return img[y0:y0 + int(h * H), x0:x0 + int(w * W)], (x0, y0)

def _tile_mask(img, s_max=None, v_min=None):
    s_max = config.TILE_S_MAX if s_max is None else s_max
    v_min = config.TILE_V_MIN if v_min is None else v_min
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # fill pips so tile is one blob
    return m

def _candidates(crop, ox, oy, area_rng, ratio_rng, tier):
    """Return list of dicts describing every blob and whether it looks like a tile."""
    mask = _tile_mask(crop, *tier)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), _ = rect
        ratio = max(w, h) / max(1.0, min(w, h))
        fill = area / max(1.0, w * h)
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        solid = cv2.countNonZero(cv2.bitwise_and(mask, filled)) / max(1.0, float(cv2.countNonZero(filled)))
        reason = None
        if area < area_rng[0]:
            reason = f"area {area:.0f} < {area_rng[0]:.0f}"
        elif area > area_rng[1]:
            reason = f"area {area:.0f} > {area_rng[1]:.0f}"
        elif not (ratio_rng[0] <= ratio <= ratio_rng[1]):
            reason = f"ratio {ratio:.2f}"
        elif fill < MIN_FILL:
            reason = f"fill {fill:.2f}"
        elif solid < MIN_SOLID:
            reason = f"solid {solid:.2f}"
        box = (cv2.boxPoints(rect) + (ox, oy)).astype(np.int32)
        out.append({"c": c, "box": box, "reason": reason, "area": area, "ratio": ratio, "fill": fill,
                    "solid": solid, "ok": reason is None, "mask": mask, "ox": ox, "oy": oy})
    return out

def _record(cands, log):
    if log is None:
        return
    for k in cands:
        log.append({"box": k["box"], "ok": k["ok"], "text": k["reason"] or "ok"})

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

def find_hand(img_small, log=None):
    crop, (ox, oy) = _crop(img_small, config.HAND_REGION)
    for tier in MASK_TIERS:
        cands = _candidates(crop, ox, oy, HAND_AREA, HAND_RATIO, tier)
        _record(cands, log)
        good = [k for k in cands if k["ok"]]
        # prefer solid, well-filled blobs; ties go to the bigger one
        for k in sorted(good, key=lambda k: -(k["solid"] + k["fill"]) * k["area"]):
            t = read_tile(crop, k["c"])
            if t:
                a, b, (cx, cy) = t
                return {"vals": (a, b), "pos": (cx + ox, cy + oy)}
    return None

def find_board(img_small, log=None):
    crop, (ox, oy) = _crop(img_small, config.BOARD_REGION)
    for tier in MASK_TIERS:
        cands = _candidates(crop, ox, oy, BOARD_AREA, BOARD_RATIO, tier)
        _record(cands, log)
        tiles = []
        for k in cands:
            if not k["ok"]:
                continue
            t = read_tile(crop, k["c"])
            if t:
                a, b, (tx, ty) = t
                tiles.append({"vals": (a, b), "pos": (tx + ox, ty + oy)})
        if tiles:
            return tiles
    return []

def debug_dump(img_small, log, hand, tiles, mask_path, out_path):
    """Write the cream mask and an image with every candidate blob and its verdict."""
    cv2.imwrite(mask_path, _tile_mask(img_small))
    out = img_small.copy()
    H, W = out.shape[:2]
    for region, col in ((config.BOARD_REGION, (255, 255, 0)), (config.HAND_REGION, (255, 0, 255))):
        x, y, w, h = region
        cv2.rectangle(out, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), col, 1)
    for k in log:
        col = (0, 200, 0) if k["ok"] else (0, 0, 255)
        cv2.polylines(out, [k["box"]], True, col, 1)
        p = k["box"][0]
        cv2.putText(out, k["text"], (int(p[0]), int(p[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.imwrite(out_path, out)
    annotate(out, hand, tiles, out_path)

def annotate(img_small, hand, tiles, path):
    out = img_small.copy()
    for t in tiles:
        cv2.putText(out, f"{t['vals'][0]}|{t['vals'][1]}", (int(t["pos"][0]) - 20, int(t["pos"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if hand:
        cv2.putText(out, f"H {hand['vals'][0]}|{hand['vals'][1]}", (int(hand["pos"][0]) - 30, int(hand["pos"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(path, out)
