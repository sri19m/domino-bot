import argparse, time, os, cv2, numpy as np
import config, vision, strategy, screens
from device import Device

def tap_px(dev, x, y, rot, orig_w):
    """Tap a pixel given in the normalized (landscape) full-res frame."""
    dev.tap_px(*vision.unrotate_point(x, y, rot, orig_w))

def once(img, name_note=""):
    """Analyse one capture, print results, write debug images. Returns (hand, tiles)."""
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/raw.png", img)
    h, w = img.shape[:2]
    print(f"capture: {w}x{h} ({'portrait' if h > w else 'landscape'})")
    img, rot = vision.normalize(img)
    if rot:
        print("portrait capture: rotated to landscape for detection (taps are mapped back)")
    small, _ = vision.downscale(img)
    log = []
    hand = vision.find_hand(small, log)
    tiles = vision.find_board(small, log)
    print("hand:", hand and hand["vals"], "board:", [t["vals"] for t in tiles])
    vision.debug_dump(small, log, hand, tiles, "debug/mask.png", "debug/once.png")
    print(f"{len(log)} candidate blobs; rejected ones are red in debug/once.png with the reason")
    if hand is None or not tiles:
        print("Nothing found: open debug/raw.png and debug/once.png. If the game is not visible in raw.png, "
              "the capture is wrong. Otherwise adjust regions/TILE_S_MAX/TILE_V_MIN in config.py.")
    return hand, tiles

POLL_EVERY = 0.3      # seconds between re-reads when no gameplay is visible
POLL_TOTAL = 4.0      # how long a frame may look "empty" before it counts as an unknown screen

class Bot:
    """Decision loop. Gameplay frames never reach screens.match or screens.teach."""
    def __init__(self, dev, debug=False, match=screens.match, teach=screens.teach,
                 sleep=time.sleep, clock=time.monotonic):
        self.dev, self.debug, self.match, self.teach = dev, debug, match, teach
        self.sleep, self.clock = sleep, clock
        self.last_sig, self.same, self.drew_for_stuck, self.empty_reads, self.n = None, 0, False, 0, 0

    def analyse(self, raw):
        img, rot = vision.normalize(raw)
        small, s = vision.downscale(img)
        hand = vision.find_hand(small)
        tiles = vision.find_board(small)
        return dict(raw=raw, img=img, rot=rot, small=small, s=s, hand=hand, tiles=tiles)

    @staticmethod
    def is_gameplay(a):
        return a["hand"] is not None or len(a["tiles"]) > 0

    def tap_frac_game(self, a, fx, fy):
        tap_px(self.dev, fx * a["img"].shape[1], fy * a["img"].shape[0], a["rot"], a["raw"].shape[1])

    def draw(self, a):
        self.tap_frac_game(a, *config.DECK_TAP)

    def gameplay(self, a):
        """Act on a gameplay frame. Returns False if the frame is stuck (treat as non-gameplay)."""
        sig = cv2.resize(cv2.cvtColor(a["small"], cv2.COLOR_BGR2GRAY), (64, 32)).astype(np.int16)
        unchanged = self.last_sig is not None and np.abs(sig - self.last_sig).mean() < 1.0
        self.last_sig = sig
        self.same = self.same + 1 if unchanged else 0
        if not unchanged:
            self.drew_for_stuck = False
        if self.same >= config.STUCK_FRAMES:
            self.same = 0
            if self.drew_for_stuck:
                print("Board did not change even after drawing; treating as a non-gameplay screen.")
                self.drew_for_stuck = False
                self.last_sig = None
                return False
            print("Board not changing: drawing from the deck.")
            self.draw(a)
            self.drew_for_stuck = True
            self.sleep(config.TAP_SETTLE)
            return True
        hand, tiles = a["hand"], a["tiles"]
        if hand is None:  # tiles visible, hand tile missing: animation. Wait and re-read.
            self.sleep(config.TAP_SETTLE)
            return True
        if self.debug:
            vision.annotate(a["small"], hand, tiles, f"debug/{self.n:04d}.png"); self.n += 1
        i = strategy.choose(hand["vals"], tiles) if tiles else None
        if i is not None:
            self.empty_reads = 0
            x, y = tiles[i]["pos"]
            tap_px(self.dev, x / a["s"], y / a["s"], a["rot"], a["raw"].shape[1])
            print(f"hand {hand['vals']}: tap {tiles[i]['vals']}")
        elif not tiles and self.empty_reads < 1:
            self.empty_reads += 1  # tiles may be mid-flip: re-read once before drawing
        else:
            self.empty_reads = 0
            self.draw(a)
            print(f"hand {hand['vals']}: " + ("no match -> draw" if tiles else "no board tiles -> draw"))
        self.sleep(config.TAP_SETTLE)
        return True

    def step(self):
        """One iteration. Returns 'gameplay', 'known', 'taught' or 'skipped' (for tests)."""
        a = self.analyse(self.dev.screenshot())
        if self.is_gameplay(a):
            if self.gameplay(a):
                return "gameplay"
        else:
            # transient frame (animation/transition): poll before deciding it is unknown
            start = self.clock()
            while self.clock() - start < POLL_TOTAL:
                self.sleep(POLL_EVERY)
                a = self.analyse(self.dev.screenshot())
                if self.is_gameplay(a):
                    return "gameplay" if self.gameplay(a) else self._non_gameplay(a)
        return self._non_gameplay(a)

    def _non_gameplay(self, a):
        known = self.match(a["raw"])
        if known:
            print(f"screen: {known['name']} -> tap")
            self.dev.tap_frac(*known["tap"])
            self.sleep(1.0)
            return "known"
        os.makedirs("debug", exist_ok=True)
        path = os.path.join("debug", f"unknown_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, a["raw"])
        print(f"Unknown screen. Frame saved to {path}")
        if self.teach(a["raw"]):
            return "taught"
        print("Nothing taught; polling again.")
        self.sleep(1.0)
        return "skipped"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--debug", action="store_true", help="save annotated frames to ./debug")
    ap.add_argument("--image", help="run --once style detection on a saved screenshot, no phone needed")
    ap.add_argument("--once", action="store_true", help="analyse one frame, print tiles, no tapping")
    args = ap.parse_args()
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read image: {args.image}")
        once(img)
        return
    dev = Device(args.serial)
    if args.debug or args.once:
        os.makedirs("debug", exist_ok=True)
    if args.once:
        once(dev.screenshot())
        return
    bot = Bot(dev, debug=args.debug)
    while True:
        bot.step()

if __name__ == "__main__":
    main()
