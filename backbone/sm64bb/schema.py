"""Column layout of the extractor's arrays (data/schema.json, written by
`dataset schema`) and the fixed lookup tables the model uses: behaviour ->
object category, surface type -> geometry class, RNG seed -> RNG index."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "..", "data", "schema.json")


class Schema:
    def __init__(self, path=SCHEMA_PATH):
        s = json.load(open(path))
        self.raw = s
        self.gi = {c: k for k, c in enumerate(s["int_cols"])}
        self.gf = {c: k for k, c in enumerate(s["flt_cols"])}
        self.oi = {c: k for k, c in enumerate(s["obj_int_cols"])}
        self.of = {c: k for k, c in enumerate(s["obj_flt_cols"])}
        self.tri = {c: k for k, c in enumerate(s["tri_cols"])}
        self.events = s["events"]
        self.behaviours = s["behaviours"]
        self.max_obj = s["max_obj"]
        self.max_dsurf = s["max_dsurf"]
        self.n_gi, self.n_gf = len(self.gi), len(self.gf)
        self.n_oi, self.n_of, self.n_tri = len(self.oi), len(self.of), len(self.tri)
        self.n_bhv = len(self.behaviours) + 1  # + unknown


# Object categories (sm64_AI_backbone.md 2.4). First match wins.
CATEGORIES = ["COLLECTIBLE", "ENEMY", "BOSS", "PLATFORM", "HAZARD", "CARRY", "INTERACT", "NPC_DIALOG", "OTHER"]
# A pattern starting with "=" matches the whole name.
_RULES = [
    ("INTERACT", ["Warp", "Door", "KoopaShell"]),
    ("COLLECTIBLE", ["BowserKey"]),
    ("BOSS", ["KingBobomb", "WhompKingBoss", "BigBoo", "GhostHuntBoo", "BalconyBigBoo", "MerryGoRoundBigBoo",
              "Eyerok", "Wiggler", "BigChillBully", "BigBully", "bhvBowser", "BowserBomb", "BowserTail",
              "BowserBody", "BigBobomb"]),
    ("NPC_DIALOG", ["Toad", "BobombBuddy", "RacingPenguin", "Ukiki", "TuxiesMother", "MessagePanel", "Yoshi",
                    "Lakitu", "Peach", "KoopaRace", "SignOnWall", "Snowman", "Message"]),
    ("CARRY", ["BreakableBoxSmall", "SmallPenguin", "Mips", "JumpingBox", "ChuckyaAnchor", "HeldObject",
               "PenguinBaby"]),
    ("COLLECTIBLE", ["Coin", "1Up", "OneUp", "Star", "Key", "WingCap", "MetalCap", "VanishCap", "NormalCap",
                     "RecoveryHeart", "Secret"]),
    ("HAZARD", ["Flamethrower", "Flame", "Fire", "Lava", "BulletBill", "BowlingBall", "ChainChomp", "Thwomp",
                "Whomp", "Grindel", "Spindel", "ToxBox", "RollingLog", "Boulder", "Tweester", "Whirlpool", "Wind",
                "Bomb", "Explosion", "ShockWave", "MrIParticle", "SnufitBalls", "SushiShark", "FallingPillar",
                "Cloud", "Spike", "Bubba", "Mine", "Snowball", "SnowmansBottom", "Pillar"]),
    ("ENEMY", ["Goomba", "Bobomb", "Koopa", "Bully", "Chuckya", "PiranhaPlant", "Spiny", "Scuttlebug", "FlyGuy",
               "Snufit", "Skeeter", "MrI", "Boo", "MontyMole", "Amp", "HeaveHo", "Klepto", "Pokey", "Moneybag",
               "Swoop", "MrBlizzard", "=bhvBub", "Spindrift", "Unagi", "CheepCheep", "Enemy", "Monkey", "Bat"]),
    ("INTERACT", ["Door", "Warp", "Pipe", "Cannon", "Pole", "Tree", "Hoot", "ExclamationBox", "CapSwitch",
                  "FloorSwitch", "BreakableBox", "KoopaShell", "Ring", "Diamond", "WoodenPost", "Chest", "Switch",
                  "Painting", "Cage", "Hidden", "Button", "Trigger"]),
    ("PLATFORM", ["Platform", "Elevator", "Lift", "TTC", "Seesaw", "Sinking", "Tilting", "Rotating", "Moving",
                  "Floating", "Bridge", "Plank", "Ferris", "Pyramid", "Block", "Carousel", "Spinner", "Pendulum",
                  "Treadmill", "Cog", "Bar", "Wood", "Rock", "Arrow", "Lily", "Raft", "Ship", "Mesh", "Flipswitch"]),
]


def category_of(name):
    for cat, parts in _RULES:
        if any(name == p[1:] if p.startswith("=") else p in name for p in parts):
            return CATEGORIES.index(cat)
    return CATEGORIES.index("OTHER")


def bhv_category_table(schema):
    t = np.array([category_of(n) for n in schema.behaviours] + [CATEGORIES.index("OTHER")], np.int64)
    return t


# Geometry kinds (sm64_AI_backbone.md 2.5): FLOOR, WALL, CEIL by normal; SPECIAL
# by surface type (hazards, quicksand, warps, friction, water/wind, misc).
GEO_KINDS = ["FLOOR", "WALL", "CEIL", "SPECIAL"]
SPECIAL_TYPES = set(
    [0x01, 0x0A]  # burning, death plane
    + [0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2D]  # quicksand
    + [0x1B, 0x1C, 0x1D, 0x1E, 0x2F, 0x32, 0xFD] + list(range(0xA6, 0xFD))  # warps, paintings
    + [0x09, 0x13, 0x14, 0x15, 0x2A, 0x2E, 0x35, 0x36, 0x37, 0x73, 0x74, 0x75, 0x78, 0x79]  # friction
    + [0x0D, 0x0E, 0x2C, 0x38]  # water, wind
    + [0x05, 0x16, 0x7B, 0x12, 0x7A, 0x33, 0x34, 0x68]  # hangable, vines, vanish walls, intangible, switch, timer
)
CAMERA_TYPES = set([0x0B, 0x65, 0x66, 0x69, 0x6E, 0x6F, 0x70, 0x72, 0x76, 0x77])


def special_table():
    t = np.zeros(257, np.int64)
    for s in SPECIAL_TYPES:
        t[s] = 1
    return t


def camera_surface_table():
    t = np.zeros(257, np.int64)
    for s in CAMERA_TYPES:
        t[s] = 1
    return t


def _random_u16(seed):
    # random_u16, src/engine/behavior_script.c.
    if seed == 0x560A:
        seed = 0
    t1 = ((seed & 0xFF) << 8) ^ seed
    seed = ((t1 & 0xFF) << 8) | ((t1 & 0xFF00) >> 8)
    t1 = ((t1 & 0xFF) << 1) ^ seed
    t2 = (t1 >> 1) ^ 0xFF80
    if t1 & 1 == 0:
        seed = 0 if t2 == 0xAA55 else t2 ^ 0x1FF4
    else:
        seed = t2 ^ 0x8180
    return seed & 0xFFFF


def rng_index_table():
    """seed -> index in the RNG's cycle from seed 0 (65114 long), normalised
    to [0, 1); seeds off the cycle map to -1."""
    idx = np.full(65536, -1.0, np.float32)
    s, k = 0, 0
    while idx[s] < 0:
        idx[s] = k
        s = _random_u16(s)
        k += 1
    n = k
    idx[idx >= 0] /= n
    return idx, n


# N64 buttons kept for pads (14 bits): A B Z Start, D-pad, L R, C buttons.
BUTTON_BITS = [15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0]
N_ACTION_ID = 512
