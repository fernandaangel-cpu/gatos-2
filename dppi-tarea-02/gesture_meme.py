"""
Webcam gesture -> Tarot meme detector (desktop version).

Opens two windows, side by side:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the Tarot card meme matching whatever gesture you're making

Hierarchy of gestures (cartas):
  1. Muerte    -> Cabeza inclinada lateralmente >= 20°, oreja al hombro (muerte.png)
  2. Sol       -> Ambos brazos levantados, manos sobre la cabeza y puntas de los dedos enfrentadas formando un arco (sol.jpeg)
  3. Mago      -> Un brazo extendido verticalmente hacia arriba, mano sobre la cabeza (Mago.jpeg)
  4. Amantes   -> Ambas manos frente al pecho formando un corazón (Amantes.jpeg)
  5. Diablo    -> Ambas manos junto a la cabeza con índices hacia arriba / cuernos (Diablo.jpeg)
  6. El Loco   -> Ambos índices apuntando a las sienes (Elloco.jpeg)
  7. Emperador -> Cabeza frontal, brazos relajados / default (emperador.jpeg)

Press q or ESC to quit.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "cartas"

GESTURE_MEMES = {
    "muerte": ["muerte.png"],
    "sol": ["sol.jpeg"],
    "mago": ["Mago.jpeg"],
    "amantes": ["Amantes.jpeg"],
    "diablo": ["Diablo.jpeg"],
    "elLoco": ["Elloco.jpeg"],
    "emperador": ["emperador.jpeg"],
}

# gestures whose meme is a video, not a still image
VIDEO_GESTURES = set()

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# Head lateral tilt angle (roll, in degrees: ear to shoulder)
MUERTE_ROLL_DEG = 20.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def roll_from_transform_matrix(matrix):
    """Extract the head's lateral tilt angle (roll, degrees) from
    MediaPipe's facial transformation matrix - ear tilting towards shoulder."""
    r = np.asarray(matrix)[:3, :3]
    roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(roll)


def yaw_from_transform_matrix(matrix):
    """Extract the head's left/right turn angle (yaw, degrees) from
    MediaPipe's facial transformation matrix."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "wrist": pts[0],
        "thumbTip": pts[4],
        "indexTip": pts[8],
        "middleTip": pts[12],
        "ringTip": pts[16],
        "pinkyTip": pts[20],
        "palmCenter": pts[9],
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


class GestureState:
    def __init__(self):
        self.last_face = None  # { mouthCenter, faceWidth, rightCheek, leftCheek, forehead, rollDeg, yawDeg, t }
        self.face_seen_this_frame = False
        self.last_roll_debug = 0.0
        self.last_yaw_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            forehead = p3(f[10])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            roll_deg = 0.0
            yaw_deg = 0.0
            if face_result.facial_transformation_matrixes:
                mat = face_result.facial_transformation_matrixes[0]
                roll_deg = roll_from_transform_matrix(mat)
                yaw_deg = yaw_from_transform_matrix(mat)
            else:
                dx = left_cheek[0] - right_cheek[0]
                dy = left_cheek[1] - right_cheek[1]
                roll_deg = math.degrees(math.atan2(dy, dx))

            self.last_face = {
                "mouthCenter": mouth_center,
                "faceWidth": face_width,
                "rightCheek": right_cheek,
                "leftCheek": left_cheek,
                "forehead": forehead,
                "mouthOpen": mouth_open,
                "rollDeg": roll_deg,
                "yawDeg": yaw_deg,
                "t": now,
            }
            self.last_roll_debug = roll_deg
            self.last_yaw_debug = yaw_deg
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face["t"] < FACE_STALE_MS

        # 1. Muerte — Cabeza inclinada lateralmente ≥20°, con una oreja acercándose al hombro
        if face_is_fresh and abs(self.last_face["rollDeg"]) >= MUERTE_ROLL_DEG:
            return "muerte"

        if not hand_result.hand_landmarks:
            return "emperador"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]
        mouth_center = self.last_face["mouthCenter"] if face_is_fresh else np.array([0.5, 0.5, 0.0])
        face_width = self.last_face["faceWidth"] if face_is_fresh else 0.2
        head_top_y = (mouth_center[1] - face_width * 0.9) if face_is_fresh else 0.35

        # Gestos con dos manos
        if len(hands) == 2:
            avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2

            # 2. Sol — Ambos brazos levantados, con las manos sobre la cabeza y las puntas de los dedos enfrentadas formando un arco
            sol_index_gap = dist(hands[0]["indexTip"], hands[1]["indexTip"]) / avg_scale
            sol_thumb_gap = dist(hands[0]["thumbTip"], hands[1]["thumbTip"]) / avg_scale
            sol_middle_gap = dist(hands[0]["middleTip"], hands[1]["middleTip"]) / avg_scale
            wrist_gap = dist(hands[0]["wrist"], hands[1]["wrist"])
            palm_gap = dist(hands[0]["palmCenter"], hands[1]["palmCenter"])
            index_dist = dist(hands[0]["indexTip"], hands[1]["indexTip"])

            both_hands_above_head = (
                hands[0]["palmCenter"][1] < head_top_y + 0.15 and
                hands[1]["palmCenter"][1] < head_top_y + 0.15
            )

            is_sol_pose = (
                both_hands_above_head and
                hands[0]["curledCount"] <= 2 and hands[1]["curledCount"] <= 2 and
                (sol_index_gap < 2.8 or sol_middle_gap < 2.8 or sol_thumb_gap < 3.2) and
                wrist_gap > index_dist * 1.05
            )

            # 4. Amantes — Ambas manos frente al pecho, índices y pulgares unidos formando un corazón
            heart_index_gap = dist(hands[0]["indexTip"], hands[1]["indexTip"]) / avg_scale
            heart_thumb_gap = dist(hands[0]["thumbTip"], hands[1]["thumbTip"]) / avg_scale
            is_amantes_pose = (
                heart_index_gap < 1.6 and heart_thumb_gap < 1.8 and
                wrist_gap > index_dist * 1.15 and
                (hands[0]["palmCenter"][1] > mouth_center[1] - 0.1 and hands[1]["palmCenter"][1] > mouth_center[1] - 0.1)
            )

            # 5. Diablo — Ambas manos junto a la parte superior de la cabeza, con ambos índices extendidos hacia arriba
            both_hands_at_head = (
                hands[0]["palmCenter"][1] < mouth_center[1] + 0.1 and
                hands[1]["palmCenter"][1] < mouth_center[1] + 0.1
            )
            both_index_up = hands[0]["indexUp"] and hands[1]["indexUp"]
            index_pointing_up = (
                hands[0]["indexTip"][1] < hands[0]["palmCenter"][1] and
                hands[1]["indexTip"][1] < hands[1]["palmCenter"][1]
            )
            other_fingers_curled = hands[0]["curledCount"] >= 2 and hands[1]["curledCount"] >= 2
            hands_spread_head = palm_gap / face_width > 0.6
            is_diablo_pose = (
                both_hands_at_head and both_index_up and index_pointing_up and
                other_fingers_curled and hands_spread_head
            )

            # 6. El Loco — Ambos índices extendidos apuntando hacia las sienes, uno a cada lado de la cabeza
            right_cheek = self.last_face["rightCheek"] if face_is_fresh else np.array([0.4, 0.4, 0.0])
            left_cheek = self.last_face["leftCheek"] if face_is_fresh else np.array([0.6, 0.4, 0.0])
            d1_r = dist(hands[0]["indexTip"], right_cheek) / face_width
            d1_l = dist(hands[0]["indexTip"], left_cheek) / face_width
            d2_r = dist(hands[1]["indexTip"], right_cheek) / face_width
            d2_l = dist(hands[1]["indexTip"], left_cheek) / face_width
            near_temples = (d1_r < 1.2 and d2_l < 1.2) or (d1_l < 1.2 and d2_r < 1.2)
            is_loco_pose = (
                both_index_up and other_fingers_curled and near_temples and
                not index_pointing_up
            )

            # Evaluar según jerarquía: Sol -> Mago -> Amantes -> Diablo -> El Loco
            if is_sol_pose and not is_amantes_pose:
                return "sol"

            # 3. Mago con 2 manos (un brazo arriba por encima de la cabeza y el otro abajo)
            hands_above_head = [
                h for h in hands if h["palmCenter"][1] < head_top_y and h["wrist"][1] > h["palmCenter"][1]
            ]
            if len(hands_above_head) == 1:
                other_hand = [h for h in hands if h is not hands_above_head[0]][0]
                if other_hand["palmCenter"][1] >= head_top_y:
                    return "mago"

            if is_amantes_pose:
                return "amantes"

            if is_diablo_pose:
                return "diablo"

            if is_loco_pose:
                return "elLoco"

        # Gestos con 1 mano visible
        if len(hands) == 1:
            h = hands[0]
            # 3. Mago — Un brazo extendido verticalmente hacia arriba, con la mano por encima de la cabeza
            if h["palmCenter"][1] < head_top_y and h["wrist"][1] > h["palmCenter"][1]:
                return "mago"

        # 7. Emperador — Cabeza frontal, manos abajo y brazos relajados a ambos lados del cuerpo (Default)
        return "emperador"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            continue
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    lines = [
        f"Carta: {gesture}",
        f"Inclinacion lateral (Roll): {state.last_roll_debug:+.1f} deg  (Muerte thr +/-{MUERTE_ROLL_DEG:.1f})",
    ]
    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "emperador"
    candidate_gesture = "emperador"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["emperador"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_meme = random.choice(memes[gesture])

            if gesture != "emperador":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "emperador":
                current_gesture = "emperador"
                current_meme = random.choice(memes["emperador"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
