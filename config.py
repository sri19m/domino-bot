"""Tunable settings. All regions are fractions of the screen (x, y, w, h), so they work at any resolution."""

# Matching rule between the hand tile and a board tile.
#   "share_end" : tile matches if any of its halves equals any half of the hand tile
#   "exact"     : tile must be the identical domino (same pair of numbers)
#   "plusminus" : any half differs by exactly 1 from any half of the hand tile (solitaire style)
RULE = "share_end"
# After tapping a board tile, does it become the new hand tile?
CHAIN = True

# Landscape reference layout measured from your 1024x477 screenshots.
BOARD_REGION = (0.15, 0.12, 0.70, 0.70)   # where board tiles live
HAND_REGION = (0.58, 0.80, 0.14, 0.20)    # face-up tile at the bottom
DECK_TAP = (0.51, 0.90)                   # tap here to draw from the deck

# Timing
TAP_SETTLE = 0.45        # seconds to wait for animations after a tap
STUCK_FRAMES = 4         # identical frames in a row => treat screen as unknown
SEARCH_NODE_CAP = 20000

# Cream tile colour (HSV): low saturation, high value
TILE_S_MAX = 70
TILE_V_MIN = 190
