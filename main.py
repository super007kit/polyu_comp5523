import math
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from audio import MicrophoneStream
from intent_semantic import INTENT_ASK_BOTTLE_POSITION
from speech_listen import listen_transcribe_and_classify
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core.image import ImageFormat
from ultralytics import YOLO

# Nano model is fastest for webcam; use "yolov8s.pt" or "yolov8m.pt" for better accuracy.
model = YOLO("yolov8n.pt")

# COCO names on this checkpoint; only these labels are detected and drawn.
FILTER_LABELS = {"person", "cup", "bottle"}
class_ids = [i for i, name in model.names.items() if name in FILTER_LABELS]
CUP_CLASS_ID = next(i for i, name in model.names.items() if name == "cup")
BOTTLE_CLASS_ID = next(i for i, name in model.names.items() if name == "bottle")

THRESH_X = 40  # pixels — how close counts as “aligned” in x
THRESH_Y = 40  # pixels — how close counts as “aligned” in y

DATA_PANEL_W = 560
DATA_PANEL_H = 320  # room for metrics + optional voice answer line
DATA_WINDOW_NAME = "Hand vs bottle data"

# Webcam window: S = listen + transcribe + intent (releases mic briefly for PyAudio)
KEY_QUIT = ord("q")
KEY_SPEECH = ord("s")

_HAND_TASK_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not _HAND_TASK_PATH.is_file():
    print("Downloading hand_landmarker.task (one-time)…")
    urllib.request.urlretrieve(_HAND_TASK_URL, _HAND_TASK_PATH)

_hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_base_options.BaseOptions(model_asset_path=str(_HAND_TASK_PATH)),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(_hand_options)
_hand_landmark_style = mp_vision.drawing_styles.get_default_hand_landmarks_style()
_hand_connection_style = mp_vision.drawing_styles.get_default_hand_connections_style()

cap = cv2.VideoCapture(0)  # try 0 first; if it doesn't work, later try 1

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


def landmarks_to_pixel_box(landmarks, width: int, height: int) -> tuple[float, float, float, float]:
    """Hand AABB from MediaPipe normalized landmarks → pixel (hxmin, hymin, hxmax, hymax)."""
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def box_center(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float]:
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def offset_guidance(dx: float, dy: float) -> tuple[str, str]:
    """dx = target_x - hand_x; dy = target_y - hand_y. Left/right and up/down guidance."""
    if abs(dx) > THRESH_X:
        instruction_x = "move right" if dx > 0 else "move left"
    else:
        instruction_x = "x aligned"
    if abs(dy) > THRESH_Y:
        instruction_y = "move down" if dy > 0 else "move up"
    else:
        instruction_y = "y aligned"
    return instruction_x, instruction_y


def yolo_boxes_for_class(
    result, class_id: int
) -> list[tuple[float, float, float, float, float]]:
    """Each entry: (xmin, ymin, xmax, ymax, conf) for a single COCO class id."""
    out: list[tuple[float, float, float, float, float]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    for b in boxes:
        if int(b.cls[0]) != class_id:
            continue
        xyxy = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0])
        out.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), conf))
    return out


def hand_left_right_vs_bottle(dx: float) -> str:
    """dx = bottle_x - hand_x (from hand_target_metrics). Describes hand vs bottle in the image."""
    if abs(dx) <= THRESH_X:
        return "Left / right: ~ aligned with bottle (same column)"
    if dx > 0:
        return "Hand is on the LEFT side of the bottle"
    return "Hand is on the RIGHT side of the bottle"


def bottle_position_answer(bottle_m: dict[str, object] | None, has_hand: bool) -> str:
    """
    dx = bottle_x - hand_x in the image (x grows to the right on screen).

    For someone facing the camera, their physical left/right do not match image left/right:
    your left side appears toward the right side of the frame. So we map dx to *your*
    left/right, not screen left/right — opposite of a naive image-space reading.
    """
    if not has_hand:
        return "I need to see your hand — put your hand in the frame near the bottle."
    if bottle_m is None:
        return "I don't see a bottle in the frame right now."
    dx = float(bottle_m["dx"])
    if abs(dx) <= THRESH_X:
        return "From your view, the bottle is roughly in line with your hand (side to side)."
    # Image: dx>0 => bottle has larger x (more to the right in the frame) => your left.
    if dx > 0:
        return "From your view, the bottle is to the left of your hand."
    return "From your view, the bottle is to the right of your hand."


def make_bottle_data_panel(
    bottle_m: dict[str, object] | None,
    has_hand: bool,
    voice_answer: str | None = None,
) -> np.ndarray:
    """BGR image for a separate OpenCV window with bottle distance and left/right text."""
    panel = np.full((DATA_PANEL_H, DATA_PANEL_W, 3), 36, dtype=np.uint8)
    y = 36
    line_h = 30

    def put(line: str, scale: float = 0.62, color: tuple[int, int, int] = (235, 235, 235)) -> None:
        nonlocal y
        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_h

    put("Hand  <->  bottle", 0.85, (200, 220, 255))
    y += 6

    if not has_hand:
        put("Status: no hand detected", 0.65, (120, 120, 255))
    elif bottle_m is None:
        put("Status: no bottle detected", 0.65, (120, 120, 255))
    else:
        dist = float(bottle_m["dist"])
        dx = float(bottle_m["dx"])
        dy = float(bottle_m["dy"])
        put(f"Distance (2D): {dist:.1f} pixels")
        y += 4
        put(hand_left_right_vs_bottle(dx), 0.68, (180, 255, 180))
        y += 4
        put(f"dx (bottle - hand): {dx:+.0f} px  |  dy: {dy:+.0f} px", 0.55, (180, 180, 180))
    if voice_answer:
        y += 8
        put("Voice (where is the bottle):", 0.62, (200, 220, 255))
        # Wrap long answers: single line with smaller font if needed
        put(voice_answer[:72] + ("..." if len(voice_answer) > 72 else ""), 0.52, (160, 255, 200))
    return panel


def hand_target_metrics(
    hand_boxes: list[tuple[float, float, float, float]],
    target_boxes: list[tuple[float, float, float, float, float]],
) -> dict[str, object] | None:
    """Closest hand to the highest-conf target box; returns centers, dist, offsets, guidance."""
    if not hand_boxes or not target_boxes:
        return None
    target_boxes = sorted(target_boxes, key=lambda t: -t[4])
    txmin, tymin, txmax, tymax, _ = target_boxes[0]
    tx_c, ty_c = box_center(txmin, tymin, txmax, tymax)

    def hand_dist_sq(hb: tuple[float, float, float, float]) -> float:
        hx_c, hy_c = box_center(*hb)
        return (hx_c - tx_c) ** 2 + (hy_c - ty_c) ** 2

    hxmin, hymin, hxmax, hymax = min(hand_boxes, key=hand_dist_sq)
    hx_c, hy_c = box_center(hxmin, hymin, hxmax, hymax)
    dx, dy = tx_c - hx_c, ty_c - hy_c
    ix, iy = offset_guidance(dx, dy)
    return {
        "dist": math.hypot(dx, dy),
        "dx": dx,
        "dy": dy,
        "ix": ix,
        "iy": iy,
        "hand_pt": (int(round(hx_c)), int(round(hy_c))),
        "target_pt": (int(round(tx_c)), int(round(ty_c))),
    }


mic = MicrophoneStream()
mic.start()
# Last speech+intent from key S: (transcript or None, intent label)
_speech_last: tuple[str | None, str] | None = None
_speech_lock = threading.Lock()
_speech_thread: threading.Thread | None = None


def _speech_worker() -> None:
    global _speech_last
    mic.stop()
    try:
        result = listen_transcribe_and_classify()
    finally:
        mic.start()
    with _speech_lock:
        _speech_last = result


def _start_speech_background() -> None:
    global _speech_thread
    if _speech_thread is not None and _speech_thread.is_alive():
        print("Speech recognition already running; wait for it to finish.")
        return
    _speech_thread = threading.Thread(target=_speech_worker, daemon=True)
    _speech_thread.start()


try:
    frame_time_ms = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]

        results = model(frame, classes=class_ids, verbose=False)
        yolo_result = results[0]
        annotated = yolo_result.plot()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp_image.Image(image_format=ImageFormat.SRGB, data=rgb)
        hand_result = hand_landmarker.detect_for_video(mp_frame, frame_time_ms)
        frame_time_ms += 1

        hand_boxes: list[tuple[float, float, float, float]] = []
        for landmarks in hand_result.hand_landmarks:
            mp_vision.drawing_utils.draw_landmarks(
                annotated,
                landmarks,
                mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                _hand_landmark_style,
                _hand_connection_style,
            )
            hand_boxes.append(landmarks_to_pixel_box(landmarks, w, h))

        cup_boxes = yolo_boxes_for_class(yolo_result, CUP_CLASS_ID)
        bottle_boxes = yolo_boxes_for_class(yolo_result, BOTTLE_CLASS_ID)

        cup_m = hand_target_metrics(hand_boxes, cup_boxes) if hand_boxes else None
        bottle_m = hand_target_metrics(hand_boxes, bottle_boxes) if hand_boxes else None

        with _speech_lock:
            speech_snap = _speech_last
        speech_listening = _speech_thread is not None and _speech_thread.is_alive()

        last_intent = speech_snap[1] if speech_snap else None
        bottle_answer: str | None = None
        if last_intent == INTENT_ASK_BOTTLE_POSITION:
            bottle_answer = bottle_position_answer(bottle_m, bool(hand_boxes))

        status_lines: list[str] = []
        if speech_listening:
            status_lines.append("speech: (listening...)")
        elif speech_snap is not None:
            st, inte = speech_snap
            status_lines.append(f"speech: {st}" if st else "speech: (not understood)")
            status_lines.append(f"intent: {inte}")
            if inte == INTENT_ASK_BOTTLE_POSITION and bottle_answer is not None:
                status_lines.append(f"answer: {bottle_answer}")
        if not hand_boxes:
            status_lines.append("no hand in frame")
        else:
            if cup_m:
                status_lines.extend(
                    [
                        f"cup dist: {cup_m['dist']:.1f} px",
                        f"cup dx={cup_m['dx']:+.0f} dy={cup_m['dy']:+.0f}",
                        f"cup: {cup_m['ix']} | {cup_m['iy']}",
                    ]
                )
            else:
                status_lines.append("no cup in frame")
            if bottle_m:
                status_lines.extend(
                    [
                        f"bottle dist: {bottle_m['dist']:.1f} px",
                        f"bottle dx={bottle_m['dx']:+.0f} dy={bottle_m['dy']:+.0f}",
                        f"bottle: {bottle_m['ix']} | {bottle_m['iy']}",
                    ]
                )
            else:
                status_lines.append("no bottle in frame")

        if cup_m:
            cv2.line(annotated, cup_m["hand_pt"], cup_m["target_pt"], (0, 255, 255), 2)
            cv2.circle(annotated, cup_m["hand_pt"], 8, (255, 128, 0), -1)
            cv2.circle(annotated, cup_m["target_pt"], 8, (0, 255, 0), -1)
        if bottle_m:
            cv2.line(annotated, bottle_m["hand_pt"], bottle_m["target_pt"], (255, 0, 255), 2)
            cv2.circle(annotated, bottle_m["hand_pt"], 8, (255, 128, 0), -1)
            cv2.circle(annotated, bottle_m["target_pt"], 8, (255, 128, 255), -1)

        y0 = 28
        for line in status_lines:
            cv2.putText(
                annotated,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0 += 26

        hint = "S=speech  Q=quit"
        hb = h - 14
        cv2.putText(annotated, hint, (10, hb), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, hint, (10, hb), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

        cv2.imshow("Webcam", annotated)
        data_panel = make_bottle_data_panel(bottle_m, bool(hand_boxes), voice_answer=bottle_answer)
        cv2.imshow(DATA_WINDOW_NAME, data_panel)

        key = cv2.waitKey(1) & 0xFF
        if key == KEY_QUIT:
            break
        if key == KEY_SPEECH:
            _start_speech_background()
finally:
    if _speech_thread is not None and _speech_thread.is_alive():
        _speech_thread.join(timeout=180.0)
    mic.stop()
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
