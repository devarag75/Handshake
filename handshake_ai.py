import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

CAMERA_URL = "http://10.199.101.175:8080/video"

MODEL_FILE = "handshake_detector.pkl"

POINTS = 40

WINDOW_SIZE = 40

PREDICTION_INTERVAL = 5


# ============================================================
# LOAD MODEL
# ============================================================

with open(
    MODEL_FILE,
    "rb"
) as file:

    model_data = pickle.load(
        file
    )


model = model_data["model"]

POINTS = model_data["points"]


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

    model_complexity=1

)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_URL
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    raise SystemExit


# ============================================================
# FUNCTIONS
# ============================================================

def get_palm_center(
    hand_landmarks,
    width,
    height
):

    indices = [
        0,
        5,
        9,
        13,
        17
    ]

    points = []

    for index in indices:

        landmark = hand_landmarks.landmark[index]

        points.append((

            int(
                landmark.x * width
            ),

            int(
                landmark.y * height
            )

        ))


    return (

        int(
            np.mean(
                [p[0] for p in points]
            )
        ),

        int(
            np.mean(
                [p[1] for p in points]
            )
        )

    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def resample_trajectory(
    trajectory,
    points=40
):

    old_x = np.linspace(
        0,
        1,
        len(trajectory)
    )

    new_x = np.linspace(
        0,
        1,
        points
    )


    x = np.interp(

        new_x,

        old_x,

        trajectory[:, 0]

    )


    y = np.interp(

        new_x,

        old_x,

        trajectory[:, 1]

    )


    return np.column_stack([
        x,
        y
    ])


# ============================================================

def extract_features(
    trajectory
):

    trajectory = resample_trajectory(

        trajectory,

        POINTS

    )


    trajectory = (
        trajectory
        -
        trajectory[0]
    )


    x_range = (

        np.max(trajectory[:, 0])
        -
        np.min(trajectory[:, 0])

    )


    y_range = (

        np.max(trajectory[:, 1])
        -
        np.min(trajectory[:, 1])

    )


    scale = max(

        x_range,

        y_range,

        1

    )


    trajectory = (
        trajectory / scale
    )


    velocity = np.diff(

        trajectory,

        axis=0

    )


    speed = np.linalg.norm(

        velocity,

        axis=1

    )


    angles = np.arctan2(

        velocity[:, 1],

        velocity[:, 0]

    )


    angle_difference = np.diff(
        angles
    )


    angle_difference = (

        (angle_difference + np.pi)

        %

        (2 * np.pi)

    ) - np.pi


    direction_changes = np.sum(

        np.abs(
            angle_difference
        ) > 0.5

    )


    acceleration = np.diff(
        speed
    )


    path_length = np.sum(
        speed
    )


    displacement = np.linalg.norm(

        trajectory[-1]
        -
        trajectory[0]

    )


    features = np.concatenate([

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

            x_range / (
                y_range + 0.001
            )

        ]

    ])


    return features


# ============================================================
# VARIABLES
# ============================================================

trajectory = deque(
    maxlen=WINDOW_SIZE
)

prediction = "WAITING"

confidence = 0.0

frame_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame = cv2.flip(
        frame,
        1
    )


    frame = cv2.resize(
        frame,
        (1280, 720)
    )


    height, width, _ = frame.shape


    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    results = hands.process(
        rgb
    )


    current_hands = []


    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS

            )


            center = get_palm_center(

                hand_landmarks,

                width,

                height

            )


            current_hands.append(
                center
            )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    current_point = None


    if len(current_hands) >= 2:

        p1 = current_hands[0]

        p2 = current_hands[1]


        midpoint = (

            int(
                (p1[0] + p2[0]) / 2
            ),

            int(
                (p1[1] + p2[1]) / 2
            )

        )


        current_point = midpoint


        cv2.line(

            frame,

            p1,

            p2,

            (255, 255, 0),

            2

        )


        cv2.circle(

            frame,

            midpoint,

            8,

            (0, 0, 255),

            -1

        )


    elif len(current_hands) == 1:

        current_point = current_hands[0]


    # ========================================================
    # ADD POINT
    # ========================================================

    if current_point is not None:

        trajectory.append(
            current_point
        )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    frame_count += 1


    if (

        len(trajectory) >= POINTS

        and

        frame_count
        % PREDICTION_INTERVAL == 0

    ):

        sequence = np.array(
            list(trajectory),
            dtype=np.float32
        )


        features = extract_features(
            sequence
        )


        prediction = model.predict(

            [features]

        )[0]


        probabilities = model.predict_proba(

            [features]

        )[0]


        confidence = np.max(
            probabilities
        )


    # ========================================================
    # DRAW TRAJECTORY
    # ========================================================

    points = list(
        trajectory
    )


    for i in range(
        1,
        len(points)
    ):

        cv2.line(

            frame,

            points[i - 1],

            points[i],

            (255, 0, 255),

            3

        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(

        frame,

        (20, 20),

        (520, 170),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        frame,

        "AI HANDSHAKE DETECTOR",

        (40, 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"RESULT: {prediction}",

        (40, 95),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (0, 255, 0),

        2

    )


    cv2.putText(

        frame,

        f"CONFIDENCE: {confidence * 100:.1f}%",

        (40, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"HANDS: {len(current_hands)}",

        (40, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    cv2.imshow(

        "AI Handshake Detector",

        frame

    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()