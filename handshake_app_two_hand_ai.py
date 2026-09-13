
import sys
from pathlib import Path
from collections import deque

import cv2
import pickle
import numpy as np
import mediapipe as mp

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel


# ============================================================
# HANDSHAKE AI - REFERENCE UI VERSION
# ============================================================
# Put these files in the SAME folder:
#   handshake_app.py
#   handshake_reference_bg.png
#   handshake_detector.pkl
#
# The UI uses the supplied reference artwork as the full
# background. The live phone camera is placed over the
# reference camera area.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CAMERA_URL = "http://192.168.1.33:8080/video"
MODEL_FILE = BASE_DIR / "handshake_detector.pkl"
BACKGROUND_FILE = BASE_DIR / "handshake_reference_bg.png"

WINDOW_W = 1536
WINDOW_H = 1024

# Coordinates are from the 1536 x 1024 reference image.
CAMERA_RECT = (402, 469, 731, 354)

# Area containing the reference score.
SCORE_RECT = (600, 124, 336, 340)

# Area containing the reference bottom HUD.
HUD_RECT = (397, 854, 740, 114)

# Center tracking box inside the live camera.
TRACK_BOX_W = 315
TRACK_BOX_H = 245

WINDOW_SIZE = 40
PREDICTION_INTERVAL = 5

# Keep this False unless your phone stream itself needs rotation.
ROTATE_DISPLAY = False

# Short-term memory helps when MediaPipe temporarily loses one hand.
HAND_MEMORY_FRAMES = 6

# Ignore tiny landmark jitter when building the trajectory.
MIN_MOVEMENT = 2.0

# AI detection tuning. These do NOT change the trained model.
# The model still receives exactly the same 40-point feature format
# that was used during training.
MIN_TRAJECTORY_PATH = 35.0
HANDSHAKE_CONFIRMATIONS = 2
NO_HANDSHAKE_CONFIRMATIONS = 3
PREDICTION_HISTORY_SIZE = 3


# ============================================================
# COLORS
# ============================================================

CYAN = QColor(57, 245, 255)
GREEN = QColor(54, 247, 160)
WHITE = QColor(244, 248, 255)
BLUE = QColor(78, 156, 255)
ORANGE = QColor(255, 179, 78)
RED = QColor(255, 85, 116)
DARK = QColor(2, 10, 20)
MUTED = QColor(134, 168, 207)


# ============================================================
# FILE CHECKS
# ============================================================

if not BACKGROUND_FILE.exists():
    print()
    print("ERROR: handshake_reference_bg.png was not found.")
    print("Expected location:")
    print(BACKGROUND_FILE)
    print()
    sys.exit(1)

if not MODEL_FILE.exists():
    print()
    print("ERROR: handshake_detector.pkl was not found.")
    print("Expected location:")
    print(MODEL_FILE)
    print()
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading trained handshake AI...")

try:
    with open(MODEL_FILE, "rb") as file:
        model_data = pickle.load(file)
except Exception as error:
    print("ERROR: Could not load the trained model.")
    print(error)
    sys.exit(1)

try:
    model = model_data["model"]
    POINTS = int(model_data["points"])
except Exception as error:
    print("ERROR: The model file does not contain the expected data.")
    print("Expected keys: model and points")
    print(error)
    sys.exit(1)

print("AI model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
    model_complexity=1,
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def resample_trajectory(trajectory, points=40):
    old_x = np.linspace(0, 1, len(trajectory))
    new_x = np.linspace(0, 1, points)

    x = np.interp(new_x, old_x, trajectory[:, 0])
    y = np.interp(new_x, old_x, trajectory[:, 1])

    return np.column_stack([x, y])


def extract_features(trajectory):
    trajectory = resample_trajectory(trajectory, POINTS)

    trajectory = trajectory - trajectory[0]

    x_range = (
        np.max(trajectory[:, 0])
        - np.min(trajectory[:, 0])
    )

    y_range = (
        np.max(trajectory[:, 1])
        - np.min(trajectory[:, 1])
    )

    scale = max(x_range, y_range, 1)

    trajectory = trajectory / scale

    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    angles = np.arctan2(
        velocity[:, 1],
        velocity[:, 0]
    )

    angle_difference = np.diff(angles)

    angle_difference = (
        (angle_difference + np.pi)
        % (2 * np.pi)
    ) - np.pi

    direction_changes = np.sum(
        np.abs(angle_difference) > 0.5
    )

    acceleration = np.diff(speed)

    path_length = np.sum(speed)

    displacement = np.linalg.norm(
        trajectory[-1] - trajectory[0]
    )

    return np.concatenate([
        trajectory.flatten(),
        velocity.flatten(),
        [
            np.mean(speed),
            np.std(speed),
            np.max(speed),
            np.min(speed),
            path_length,
            x_range,
            y_range,
            direction_changes,
            displacement,
            np.std(acceleration),
            x_range / (y_range + 0.001),
        ],
    ])


# ============================================================
# PALM CENTER
# ============================================================

def get_palm_center(hand_landmarks, width, height):
    indices = [0, 5, 9, 13, 17]
    points = []

    for index in indices:
        landmark = hand_landmarks.landmark[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return (
        int(np.mean([p[0] for p in points])),
        int(np.mean([p[1] for p in points])),
    )


def point_distance(p1, p2):
    return float(np.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    ))


def stabilize_hands(current_hands, previous_hands, missing_frames):
    """
    Keep hand positions stable for a few frames when MediaPipe
    temporarily loses one hand during an overlap.
    """
    if len(current_hands) >= 2:
        return current_hands[:2], current_hands[:2], 0

    if len(current_hands) == 1 and len(previous_hands) == 2:
        visible = current_hands[0]

        d0 = point_distance(visible, previous_hands[0])
        d1 = point_distance(visible, previous_hands[1])

        if d0 <= d1:
            ordered = [visible, previous_hands[1]]
        else:
            ordered = [previous_hands[0], visible]

        new_missing_frames = missing_frames + 1

        if new_missing_frames <= HAND_MEMORY_FRAMES:
            return ordered, previous_hands, new_missing_frames

    if len(current_hands) == 0 and len(previous_hands) == 2:
        new_missing_frames = missing_frames + 1

        if new_missing_frames <= HAND_MEMORY_FRAMES:
            return previous_hands[:2], previous_hands[:2], new_missing_frames

    return current_hands[:2], previous_hands, missing_frames + 1


# ============================================================
# SCORE OVERLAY
# ============================================================

class ScoreOverlay(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.score = 0
        self.result = "WAITING"

        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_data(self, score, result):
        self.score = max(0, min(100, int(score)))
        self.result = str(result)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        cx = w / 2
        cy = h / 2

        # Completely hide the score printed in the reference artwork.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(2, 12, 24, 245))
        painter.drawEllipse(
            QPointF(cx, cy),
            min(w, h) * 0.39,
            min(w, h) * 0.39
        )

        # Outer rings.
        for radius, width, alpha in [
            (min(w, h) * 0.43, 1, 70),
            (min(w, h) * 0.46, 1, 120),
        ]:
            pen = QPen(QColor(57, 245, 255, alpha))
            pen.setWidth(width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                QPointF(cx, cy),
                radius,
                radius
            )

        # Main progress ring.
        if self.result == "HANDSHAKE":
            color = GREEN
        elif self.result == "NO HANDSHAKE":
            color = ORANGE
        else:
            color = CYAN

        ring_radius = min(w, h) * 0.35

        ring_pen = QPen(color)
        ring_pen.setWidth(max(8, int(min(w, h) * 0.055)))
        ring_pen.setCapStyle(Qt.RoundCap)

        painter.setPen(ring_pen)

        rect = QRectF(
            cx - ring_radius,
            cy - ring_radius,
            ring_radius * 2,
            ring_radius * 2
        )

        span = int(-360 * 16 * self.score / 100)

        if self.score > 0:
            painter.drawArc(
                rect,
                90 * 16,
                span
            )

        # Small start marker.
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)

        painter.drawEllipse(
            QPointF(cx, cy - ring_radius),
            4,
            4
        )

        # Title.
        painter.setPen(WHITE)
        painter.setFont(
            QFont("Segoe UI", max(11, int(w * 0.045)), QFont.Medium)
        )

        painter.drawText(
            QRectF(cx - w * 0.30, cy - h * 0.22,
                   w * 0.60, h * 0.10),
            Qt.AlignCenter,
            "Handshake"
        )

        painter.drawText(
            QRectF(cx - w * 0.30, cy - h * 0.13,
                   w * 0.60, h * 0.10),
            Qt.AlignCenter,
            "Score"
        )

        # Score number.
        painter.setFont(
            QFont("Segoe UI", max(30, int(w * 0.13)), QFont.Bold)
        )

        painter.drawText(
            QRectF(cx - w * 0.32, cy - h * 0.01,
                   w * 0.64, h * 0.22),
            Qt.AlignCenter,
            str(self.score)
        )

        painter.setPen(MUTED)
        painter.setFont(
            QFont("Segoe UI", max(10, int(w * 0.043)), QFont.Medium)
        )

        painter.drawText(
            QRectF(cx - w * 0.20, cy + h * 0.17,
                   w * 0.40, h * 0.09),
            Qt.AlignCenter,
            "/ 100"
        )

        painter.end()


# ============================================================
# BOTTOM HUD
# ============================================================

class HudOverlay(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.score = 0
        self.hands = 0
        self.status = "READY"
        self.result = "WAITING"

        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_data(self, score, hands, status, result):
        self.score = int(score)
        self.hands = int(hands)
        self.status = str(status)
        self.result = str(result)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Hide the HUD printed in the reference.
        painter.setBrush(QColor(2, 13, 27, 245))

        border = QPen(QColor(57, 245, 255, 190))
        border.setWidth(1)

        painter.setPen(border)

        painter.drawRoundedRect(
            QRectF(1, 1, w - 2, h - 2),
            22,
            22
        )

        # Separators.
        painter.setPen(
            QPen(QColor(90, 150, 205, 100), 1)
        )

        for x in (w * 0.335, w * 0.665):
            painter.drawLine(
                QPointF(x, 16),
                QPointF(x, h - 16)
            )

        # Column helper.
        def draw_column(left, right, title, value, color, font_size):
            painter.setPen(MUTED)
            painter.setFont(
                QFont("Segoe UI", max(9, int(h * 0.10)), QFont.Medium)
            )

            painter.drawText(
                QRectF(left, h * 0.14, right - left, h * 0.22),
                Qt.AlignCenter,
                title
            )

            painter.setPen(color)
            painter.setFont(
                QFont("Segoe UI", font_size, QFont.Bold)
            )

            painter.drawText(
                QRectF(left, h * 0.38, right - left, h * 0.44),
                Qt.AlignCenter,
                value
            )

        draw_column(
            0,
            w * 0.335,
            "SCORE",
            f"{self.score} / 100",
            GREEN if self.score >= 70 else CYAN,
            max(19, int(h * 0.23))
        )

        draw_column(
            w * 0.335,
            w * 0.665,
            "HANDS DETECTED",
            f"{self.hands} / 2",
            WHITE,
            max(19, int(h * 0.23))
        )

        if self.result == "HANDSHAKE":
            status_color = GREEN
        elif self.result == "NO HANDSHAKE":
            status_color = ORANGE
        else:
            status_color = CYAN

        draw_column(
            w * 0.665,
            w,
            "STATUS",
            self.status,
            status_color,
            max(14, int(h * 0.17))
        )

        painter.end()


# ============================================================
# MAIN WINDOW
# ============================================================

class HandshakeApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handshake AI")

        self.setMinimumSize(1100, 730)
        self.resize(WINDOW_W, WINDOW_H)

        central = QWidget()
        self.setCentralWidget(central)

        central.setStyleSheet(
            "background: #020711;"
        )

        self.central = central

        # --------------------------------------------------------
        # REFERENCE BACKGROUND
        # --------------------------------------------------------

        self.background = QLabel(central)
        self.background.setAlignment(Qt.AlignCenter)
        self.background.setAttribute(
            Qt.WA_TransparentForMouseEvents
        )

        self.background_pixmap = QPixmap(
            str(BACKGROUND_FILE)
        )

        if self.background_pixmap.isNull():
            print("ERROR: Background image could not be opened.")
            sys.exit(1)

        # --------------------------------------------------------
        # LIVE CAMERA
        # --------------------------------------------------------

        self.video = QLabel(central)
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet(
            """
            QLabel {
                background: #02080F;
                border: none;
                border-radius: 12px;
            }
            """
        )

        self.video.setAttribute(
            Qt.WA_TransparentForMouseEvents
        )

        # --------------------------------------------------------
        # CAMERA LABELS
        # --------------------------------------------------------

        self.live_label = QLabel(
            "LIVE CAMERA",
            central
        )

        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setStyleSheet(
            """
            QLabel {
                color: #F4F8FF;
                background: rgba(3, 20, 34, 225);
                border: 1px solid #36F7A0;
                border-radius: 17px;
                padding: 7px 15px;
                font-family: Segoe UI;
                font-size: 12px;
                font-weight: 700;
            }
            """
        )

        self.ai_label = QLabel(
            "AI TRACKING",
            central
        )

        self.ai_label.setAlignment(Qt.AlignCenter)
        self.ai_label.setStyleSheet(
            """
            QLabel {
                color: #39F5FF;
                background: rgba(3, 20, 34, 225);
                border: 1px solid #39F5FF;
                border-radius: 17px;
                padding: 7px 15px;
                font-family: Segoe UI;
                font-size: 12px;
                font-weight: 700;
            }
            """
        )

        # --------------------------------------------------------
        # SCORE
        # --------------------------------------------------------

        self.score_overlay = ScoreOverlay(central)

        # --------------------------------------------------------
        # HUD
        # --------------------------------------------------------

        self.hud = HudOverlay(central)

        # --------------------------------------------------------
        # STATE
        # --------------------------------------------------------

        self.trajectory = deque(maxlen=WINDOW_SIZE)

        self.frame_count = 0

        self.prediction = "WAITING"
        self.confidence = 0.0
        self.score = 0
        self.hand_count = 0

        # Keep a very small history of AI decisions. This prevents one
        # unstable frame from immediately declaring a handshake.
        self.prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.last_ai_prediction = "WAITING"
        self.last_ai_confidence = 0.0

        # Short-term MediaPipe hand memory.
        self.previous_hands = []
        self.missing_frames = 0

        # --------------------------------------------------------
        # CAMERA
        # --------------------------------------------------------

        print("Connecting to phone camera...")
        print(CAMERA_URL)

        self.cap = cv2.VideoCapture(CAMERA_URL)

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.cap.isOpened():
            print()
            print("ERROR: Camera could not be opened.")
            print("Check IP Webcam and the phone connection.")
            print(CAMERA_URL)
            sys.exit(1)

        print("Camera connected successfully.")
        print("Handshake AI started.")

        # --------------------------------------------------------
        # TIMER
        # --------------------------------------------------------

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.position_layers()

    # ========================================================
    # SCALE REFERENCE IMAGE TO FILL WINDOW
    # ========================================================

    def update_background(self):
        w = self.central.width()
        h = self.central.height()

        if w <= 0 or h <= 0:
            return

        # Fill the complete window, cropping only if aspect ratio differs.
        scaled = self.background_pixmap.scaled(
            w,
            h,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )

        left = max(0, (scaled.width() - w) // 2)
        top = max(0, (scaled.height() - h) // 2)

        cropped = scaled.copy(
            left,
            top,
            w,
            h
        )

        self.background.setPixmap(cropped)
        self.background.setGeometry(0, 0, w, h)

    # ========================================================
    # POSITION ALL REFERENCE ELEMENTS
    # ========================================================

    def position_layers(self):
        cw = self.central.width()
        ch = self.central.height()

        if cw <= 0 or ch <= 0:
            return

        sx = cw / WINDOW_W
        sy = ch / WINDOW_H

        self.update_background()

        # Camera.
        x, y, w, h = CAMERA_RECT

        self.video.setGeometry(
            int(x * sx),
            int(y * sy),
            int(w * sx),
            int(h * sy)
        )

        # Camera labels.
        self.live_label.adjustSize()
        self.ai_label.adjustSize()

        self.live_label.move(
            int((x + 2) * sx),
            int((y - 48) * sy)
        )

        self.ai_label.move(
            int(
                (x + w) * sx
                - self.ai_label.width()
                - int(2 * sx)
            ),
            int((y - 48) * sy)
        )

        # Score.
        sx0, sy0, sw, sh = SCORE_RECT

        self.score_overlay.setGeometry(
            int(sx0 * sx),
            int(sy0 * sy),
            int(sw * sx),
            int(sh * sy)
        )

        # HUD.
        hx, hy, hw, hh = HUD_RECT

        self.hud.setGeometry(
            int(hx * sx),
            int(hy * sy),
            int(hw * sx),
            int(hh * sy)
        )

        self.background.lower()

        self.video.raise_()
        self.live_label.raise_()
        self.ai_label.raise_()
        self.score_overlay.raise_()
        self.hud.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_layers()

    # ========================================================
    # DISPLAY LIVE CAMERA
    # ========================================================

    def display_frame(self, frame):
        display = frame

        if ROTATE_DISPLAY:
            display = cv2.rotate(
                display,
                cv2.ROTATE_90_CLOCKWISE
            )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        target = self.video.size()

        if target.width() <= 10 or target.height() <= 10:
            return

        scaled = pixmap.scaled(
            target,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )

        left = max(
            0,
            (scaled.width() - target.width()) // 2
        )

        top = max(
            0,
            (scaled.height() - target.height()) // 2
        )

        cropped = scaled.copy(
            left,
            top,
            target.width(),
            target.height()
        )

        self.video.setPixmap(cropped)

    # ========================================================
    # CAMERA + AI
    # ========================================================

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.live_label.setText("CAMERA ERROR")
            return

        self.live_label.setText("LIVE CAMERA")

        # --------------------------------------------------------
        # AI FRAME
        # --------------------------------------------------------
        # All AI drawings are made on ai_frame, and ai_frame is
        # the exact frame sent to the Qt camera widget.
        ai_frame = cv2.flip(frame, 1)

        height, width, _ = ai_frame.shape

        rgb = cv2.cvtColor(
            ai_frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        detected_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                center = get_palm_center(
                    hand_landmarks,
                    width,
                    height
                )

                if center is not None:
                    detected_hands.append(center)

                # MediaPipe skeleton.
                mp_drawing.draw_landmarks(
                    image=ai_frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS
                )

        raw_hand_count = len(detected_hands)

        # --------------------------------------------------------
        # STABILIZE HAND DETECTION
        # --------------------------------------------------------
        current_hands, self.previous_hands, self.missing_frames = (
            stabilize_hands(
                detected_hands,
                self.previous_hands,
                self.missing_frames
            )
        )

        # HUD reports actual MediaPipe detections.
        self.hand_count = raw_hand_count

        # Palm centers.
        for p in current_hands:
            cv2.circle(
                ai_frame,
                p,
                8,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )

        # --------------------------------------------------------
        # WHITE LINE BETWEEN HANDS
        # --------------------------------------------------------
        current_point = None

        # IMPORTANT:
        # The handshake AI is allowed to build a trajectory ONLY
        # when MediaPipe currently sees TWO hands.
        #
        # This prevents one visible hand from being classified as
        # a handshake by the trained model.
        if raw_hand_count >= 2 and len(current_hands) >= 2:

            p1 = current_hands[0]
            p2 = current_hands[1]

            # Bright white connection line.
            cv2.line(
                ai_frame,
                p1,
                p2,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            midpoint = (
                int((p1[0] + p2[0]) / 2),
                int((p1[1] + p2[1]) / 2)
            )

            # Midpoint marker.
            cv2.circle(
                ai_frame,
                midpoint,
                7,
                (0, 255, 255),
                -1,
                cv2.LINE_AA
            )

            current_point = midpoint

        # --------------------------------------------------------
        # TRAJECTORY
        # --------------------------------------------------------
        if current_point is not None:

            if len(self.trajectory) == 0:

                self.trajectory.append(
                    current_point
                )

            elif point_distance(
                self.trajectory[-1],
                current_point
            ) >= MIN_MOVEMENT:

                self.trajectory.append(
                    current_point
                )

        # Draw trajectory on the same displayed frame.
        trajectory_points = list(
            self.trajectory
        )

        for i in range(
            1,
            len(trajectory_points)
        ):

            cv2.line(
                ai_frame,
                trajectory_points[i - 1],
                trajectory_points[i],
                (255, 0, 255),
                3,
                cv2.LINE_AA
            )

        # --------------------------------------------------------
        # AI PREDICTION
        # --------------------------------------------------------
        self.frame_count += 1

        # The trained AI was built for a 40-point trajectory. We therefore
        # never ask it to classify a shorter sequence.
        #
        # IMPORTANT: the AI is only fed a trajectory when TWO hands are
        # currently detected. A single hand can therefore never trigger
        # HANDSHAKE.
        if raw_hand_count < 2:

            self.trajectory.clear()
            self.prediction_history.clear()
            self.prediction = "WAITING"
            self.confidence = 0.0
            self.score = 0
            self.last_ai_prediction = "WAITING"
            self.last_ai_confidence = 0.0

        elif (
            len(self.trajectory) >= POINTS
            and self.frame_count % PREDICTION_INTERVAL == 0
        ):

            sequence = np.array(
                list(self.trajectory),
                dtype=np.float32
            )

            # Motion gate: do not classify a stationary pair of hands.
            # This is only a safety gate before the trained AI; it does not
            # replace the AI model.
            diffs = np.diff(sequence, axis=0)
            path_length_pixels = float(
                np.sum(np.linalg.norm(diffs, axis=1))
            )

            if path_length_pixels >= MIN_TRAJECTORY_PATH:

                try:

                    # EXACTLY the same feature extraction used by the
                    # training program.
                    features = extract_features(sequence)

                    prediction = model.predict(
                        [features]
                    )[0]

                    probabilities = model.predict_proba(
                        [features]
                    )[0]

                    raw_prediction = str(prediction).upper()
                    raw_confidence = float(np.max(probabilities))

                    self.last_ai_prediction = raw_prediction
                    self.last_ai_confidence = raw_confidence

                    # Store the AI result, then require short-term
                    # agreement before changing the visible result.
                    self.prediction_history.append(raw_prediction)

                    handshake_votes = sum(
                        1
                        for item in self.prediction_history
                        if item == "HANDSHAKE"
                    )

                    no_handshake_votes = sum(
                        1
                        for item in self.prediction_history
                        if item == "NO_HANDSHAKE"
                    )

                    if handshake_votes >= HANDSHAKE_CONFIRMATIONS:
                        self.prediction = "HANDSHAKE"
                        self.confidence = raw_confidence

                    elif no_handshake_votes >= NO_HANDSHAKE_CONFIRMATIONS:
                        self.prediction = "NO_HANDSHAKE"
                        self.confidence = raw_confidence

                    else:
                        # Keep the current decision while the AI gathers
                        # enough consecutive evidence.
                        if self.prediction not in (
                            "HANDSHAKE",
                            "NO_HANDSHAKE"
                        ):
                            self.prediction = "WAITING"
                            self.confidence = raw_confidence

                    prediction_upper = self.prediction.upper()

                    # ----------------------------------------------------
                    # SCORE
                    # ----------------------------------------------------
                    # Score is based on the trained model's actual
                    # HANDSHAKE probability.
                    classes = getattr(model, "classes_", [])
                    handshake_probability = 0.0

                    for index, class_value in enumerate(classes):
                        if str(class_value).upper() == "HANDSHAKE":
                            handshake_probability = float(
                                probabilities[index]
                            )
                            break

                    if prediction_upper == "HANDSHAKE":
                        self.score = int(
                            max(
                                0,
                                min(
                                    100,
                                    round(handshake_probability * 100)
                                )
                            )
                        )

                    elif prediction_upper == "NO_HANDSHAKE":
                        self.score = 0

                    elif prediction_upper in (
                        "POOR",
                        "AVERAGE",
                        "GOOD",
                        "EXCELLENT",
                    ):
                        # Compatibility with a future four-class model.
                        class_scores = {
                            "POOR": 25,
                            "AVERAGE": 50,
                            "GOOD": 75,
                            "EXCELLENT": 95,
                        }

                        weighted_score = 0.0

                        for class_name, class_score in class_scores.items():
                            for index, class_value in enumerate(classes):
                                if str(class_value).upper() == class_name:
                                    weighted_score += (
                                        float(probabilities[index])
                                        * class_score
                                    )

                        self.score = int(
                            max(
                                0,
                                min(
                                    100,
                                    round(weighted_score)
                                )
                            )
                        )

                    else:
                        self.score = 0

                except Exception as error:
                    print("Prediction error:", error)

            else:
                # Two hands are visible, but there is not enough movement
                # yet to call the trained AI.
                if self.prediction != "HANDSHAKE":
                    self.prediction = "WAITING"
                    self.confidence = 0.0
                    self.score = 0
                    self.prediction_history.clear()

        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------
        prediction_upper = (
            self.prediction.upper()
        )

        if prediction_upper == "HANDSHAKE":

            status = "DETECTED"

        elif prediction_upper == "NO_HANDSHAKE":

            status = "READY"

        elif prediction_upper == "POOR":

            status = "POOR"

        elif prediction_upper == "AVERAGE":

            status = "AVERAGE"

        elif prediction_upper == "GOOD":

            status = "GOOD"

        elif prediction_upper == "EXCELLENT":

            status = "EXCELLENT"

        else:

            status = "READY"

        # --------------------------------------------------------
        # OVERLAYS
        # --------------------------------------------------------
        self.score_overlay.set_data(
            self.score,
            prediction_upper
        )

        self.hud.set_data(
            self.score,
            self.hand_count,
            status,
            prediction_upper
        )

        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------
        # IMPORTANT:
        # Display ai_frame, not the original untouched frame.
        # This makes the white hand line, centers, skeleton and
        # trajectory visible in the live camera.
        self.display_frame(
            ai_frame
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass

        try:
            self.cap.release()
        except Exception:
            pass

        try:
            hands.close()
        except Exception:
            pass

        event.accept()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = HandshakeApp()
    window.show()

    sys.exit(app.exec())
