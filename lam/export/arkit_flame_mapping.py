import os
import json
import numpy as np


# 52 ARKit blendshape names in canonical order
ARKIT_BLENDSHAPE_NAMES = [
    "browDownLeft",         # 0
    "browDownRight",        # 1
    "browInnerUp",          # 2
    "browOuterUpLeft",      # 3
    "browOuterUpRight",     # 4
    "cheekPuff",            # 5
    "cheekSquintLeft",      # 6
    "cheekSquintRight",     # 7
    "eyeBlinkLeft",         # 8
    "eyeBlinkRight",        # 9
    "eyeLookDownLeft",      # 10
    "eyeLookDownRight",     # 11
    "eyeLookInLeft",        # 12
    "eyeLookInRight",       # 13
    "eyeLookOutLeft",       # 14
    "eyeLookOutRight",      # 15
    "eyeLookUpLeft",        # 16
    "eyeLookUpRight",       # 17
    "eyeSquintLeft",        # 18
    "eyeSquintRight",       # 19
    "eyeWideLeft",          # 20
    "eyeWideRight",         # 21
    "jawForward",           # 22
    "jawLeft",              # 23
    "jawOpen",              # 24
    "jawRight",             # 25
    "mouthClose",           # 26
    "mouthDimpleLeft",      # 27
    "mouthDimpleRight",     # 28
    "mouthFrownLeft",       # 29
    "mouthFrownRight",      # 30
    "mouthFunnel",          # 31
    "mouthLeft",            # 32
    "mouthLowerDownLeft",   # 33
    "mouthLowerDownRight",  # 34
    "mouthPressLeft",       # 35
    "mouthPressRight",      # 36
    "mouthPucker",          # 37
    "mouthRight",           # 38
    "mouthRollLower",       # 39
    "mouthRollUpper",       # 40
    "mouthShrugLower",      # 41
    "mouthShrugUpper",      # 42
    "mouthSmileLeft",       # 43
    "mouthSmileRight",      # 44
    "mouthStretchLeft",     # 45
    "mouthStretchRight",    # 46
    "mouthUpperUpLeft",     # 47
    "mouthUpperUpRight",    # 48
    "noseSneerLeft",        # 49
    "noseSneerRight",       # 50
    "tongueOut",            # 51
]


def generate_arkit_flame_mapping():
    """
    Generate a hand-tuned sparse [52, 100] mapping matrix from
    52 ARKit blendshapes to 100 FLAME expression parameters.

    The matrix M is used as: flame_expr = arkit_weights @ M
    where arkit_weights is [52] and flame_expr is [100].

    Returns:
        np.ndarray of shape [52, 100]
    """
    M = np.zeros((52, 100), dtype=np.float32)

    # Brow expressions
    # browDownLeft (0) -> FLAME brow-down-left expression components
    M[0, 1] = -0.6
    M[0, 3] = -0.3
    # browDownRight (1) -> FLAME brow-down-right
    M[1, 2] = -0.6
    M[1, 4] = -0.3
    # browInnerUp (2) -> FLAME inner brow raise
    M[2, 1] = 0.7
    M[2, 2] = 0.7
    # browOuterUpLeft (3) -> FLAME outer brow raise left
    M[3, 3] = 0.8
    # browOuterUpRight (4) -> FLAME outer brow raise right
    M[4, 4] = 0.8

    # Cheek expressions
    # cheekPuff (5) -> FLAME cheek puff
    M[5, 30] = 0.7
    M[5, 31] = 0.7
    # cheekSquintLeft (6) -> FLAME cheek squint left
    M[6, 32] = 0.6
    # cheekSquintRight (7) -> FLAME cheek squint right
    M[7, 33] = 0.6

    # Eye blink
    # eyeBlinkLeft (8) -> FLAME eye blink left
    M[8, 5] = 0.9
    # eyeBlinkRight (9) -> FLAME eye blink right
    M[9, 6] = 0.9

    # Eye look directions
    # eyeLookDownLeft (10)
    M[10, 7] = 0.5
    # eyeLookDownRight (11)
    M[11, 8] = 0.5
    # eyeLookInLeft (12)
    M[12, 9] = 0.4
    # eyeLookInRight (13)
    M[13, 10] = 0.4
    # eyeLookOutLeft (14)
    M[14, 9] = -0.4
    # eyeLookOutRight (15)
    M[15, 10] = -0.4
    # eyeLookUpLeft (16)
    M[16, 7] = -0.5
    # eyeLookUpRight (17)
    M[17, 8] = -0.5

    # Eye squint
    # eyeSquintLeft (18)
    M[18, 11] = 0.5
    # eyeSquintRight (19)
    M[19, 12] = 0.5

    # Eye wide
    # eyeWideLeft (20)
    M[20, 5] = -0.6
    M[20, 11] = -0.3
    # eyeWideRight (21)
    M[21, 6] = -0.6
    M[21, 12] = -0.3

    # Jaw expressions
    # jawForward (22)
    M[22, 13] = 0.5
    # jawLeft (23)
    M[23, 14] = 0.5
    # jawOpen (24) -> FLAME jaw-related expression component
    M[24, 15] = 0.9
    M[24, 16] = 0.3
    # jawRight (25)
    M[25, 14] = -0.5

    # Mouth expressions
    # mouthClose (26)
    M[26, 15] = -0.4
    # mouthDimpleLeft (27)
    M[27, 34] = 0.4
    # mouthDimpleRight (28)
    M[28, 35] = 0.4
    # mouthFrownLeft (29)
    M[29, 17] = 0.7
    # mouthFrownRight (30)
    M[30, 18] = 0.7
    # mouthFunnel (31) -> FLAME lip funnel
    M[31, 19] = 0.8
    M[31, 20] = 0.4
    # mouthLeft (32)
    M[32, 21] = 0.5
    # mouthLowerDownLeft (33)
    M[33, 22] = 0.5
    # mouthLowerDownRight (34)
    M[34, 23] = 0.5
    # mouthPressLeft (35)
    M[35, 24] = 0.4
    # mouthPressRight (36)
    M[36, 25] = 0.4
    # mouthPucker (37) -> FLAME lip pucker
    M[37, 20] = 0.8
    M[37, 19] = 0.3
    # mouthRight (38)
    M[38, 21] = -0.5
    # mouthRollLower (39)
    M[39, 26] = 0.5
    # mouthRollUpper (40)
    M[40, 27] = 0.5
    # mouthShrugLower (41)
    M[41, 28] = 0.5
    # mouthShrugUpper (42)
    M[42, 29] = 0.5
    # mouthSmileLeft (43) -> FLAME smile left
    M[43, 36] = 0.8
    M[43, 38] = 0.3
    # mouthSmileRight (44) -> FLAME smile right
    M[44, 37] = 0.8
    M[44, 39] = 0.3
    # mouthStretchLeft (45)
    M[45, 40] = 0.5
    # mouthStretchRight (46)
    M[46, 41] = 0.5
    # mouthUpperUpLeft (47)
    M[47, 42] = 0.5
    # mouthUpperUpRight (48)
    M[48, 43] = 0.5

    # Nose expressions
    # noseSneerLeft (49) -> FLAME nose sneer left
    M[49, 44] = 0.6
    # noseSneerRight (50) -> FLAME nose sneer right
    M[50, 45] = 0.6

    # Tongue
    # tongueOut (51)
    M[51, 46] = 0.7

    return M


def save_mapping(matrix, path):
    """
    Save an ARKit-to-FLAME mapping matrix to JSON.

    Args:
        matrix: np.ndarray of shape [52, 100]
        path: output JSON file path
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "arkit_blendshape_names": ARKIT_BLENDSHAPE_NAMES,
        "shape": list(matrix.shape),
        "matrix": matrix.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_mapping(path):
    """
    Load an ARKit-to-FLAME mapping matrix from JSON.

    Args:
        path: path to JSON file

    Returns:
        np.ndarray of shape [52, 100]
    """
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["matrix"], dtype=np.float32)


def get_default_mapping_path():
    """
    Return the path to the default ARKit-to-FLAME mapping JSON.
    Creates the file if it does not exist.

    Returns:
        str: absolute path to assets/default_arkit_to_flame.json
    """
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
    default_path = os.path.join(assets_dir, "default_arkit_to_flame.json")
    if not os.path.exists(default_path):
        os.makedirs(assets_dir, exist_ok=True)
        M = generate_arkit_flame_mapping()
        save_mapping(M, default_path)
    return default_path
