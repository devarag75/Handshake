import cv2
import mediapipe as mp
import csv
import os
import numpy as np


# ============================================================
# CAMERA
# ============================================================

CAMERA_URL = "http://10.199.101.175:8080/video"

DATASET_FILE = "handshake_detection_dataset.csv"


# ============================================================
# SETTINGS
# ============================================================

MAX_POINTS = 120

MIN_POINTS = 20

SMOOTHING = 0.70

HAND_MEMORY_FRAMES = 6


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):

    with open(
        DATASET_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "label",
            "trajectory"
        ])

    print("Created:", DATASET_FILE)


# ============================================================
# CAMERA
# ============================================================

print()
print("Connecting to phone camera...")
print()

cap = cv2.VideoCapture(CAMERA_URL)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():

    print("ERROR: Could not open phone camera.")
    print()
    print("Camera URL:")
    print(CAMERA_URL)

    input("Press ENTER to exit...")
    raise SystemExit


print("Phone camera connected.")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.35,

    min_tracking_confidence=0.35,

    model_complexity=1
)


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

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    if not points:
        return None

    x = int(
        sum(p[0] for p in points)
        / len(points)
    )

    y = int(
        sum(p[1] for p in points)
        / len(points)
    )

    return (x, y)


# ============================================================

def distance(p1, p2):

    return np.sqrt(

        (p1[0] - p2[0]) ** 2

        +

        (p1[1] - p2[1]) ** 2

    )


# ============================================================
# VARIABLES
# ============================================================

trajectory = []

recording = False

sample_number = 0

smooth_point = None

hand_memory = []

missing_frames = 0


# ============================================================
# WINDOW
# ============================================================

WINDOW = "AI Handshake Detection Dataset"

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    1280,
    720
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame error.")

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


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)


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

                if center is not None:

                    current_hands.append(
                        center
                    )


        detected_hands = len(
            current_hands
        )


        # ====================================================
        # HAND MEMORY
        # ====================================================

        if detected_hands == 2:

            hand_memory = current_hands.copy()

            missing_frames = 0


        elif detected_hands == 1:

            missing_frames += 1

            if (

                len(hand_memory) == 2

                and

                missing_frames
                <= HAND_MEMORY_FRAMES

            ):

                current_hands = [

                    current_hands[0],

                    hand_memory[1]

                ]


        else:

            missing_frames += 1

            if (

                len(hand_memory) == 2

                and

                missing_frames
                <= HAND_MEMORY_FRAMES

            ):

                current_hands = hand_memory.copy()


        stable_hands = len(
            current_hands
        )


        # ====================================================
        # DRAW CENTERS
        # ====================================================

        for p in current_hands:

            cv2.circle(

                frame,

                p,

                9,

                (0, 255, 0),

                -1

            )


        # ====================================================
        # TRAJECTORY POINT
        # ====================================================

        current_point = None


        if len(current_hands) >= 2:

            p1 = current_hands[0]

            p2 = current_hands[1]


            cv2.line(

                frame,

                p1,

                p2,

                (255, 255, 0),

                2

            )


            midpoint = (

                int(
                    (p1[0] + p2[0]) / 2
                ),

                int(
                    (p1[1] + p2[1]) / 2
                )

            )


            cv2.circle(

                frame,

                midpoint,

                8,

                (0, 0, 255),

                -1

            )


            current_point = midpoint


        elif len(current_hands) == 1:

            current_point = current_hands[0]


        # ====================================================
        # SMOOTHING
        # ====================================================

        if current_point is not None:

            if smooth_point is None:

                smooth_point = current_point

            else:

                smooth_point = (

                    int(
                        SMOOTHING
                        * smooth_point[0]

                        +

                        (1 - SMOOTHING)
                        * current_point[0]
                    ),

                    int(
                        SMOOTHING
                        * smooth_point[1]

                        +

                        (1 - SMOOTHING)
                        * current_point[1]
                    )

                )

            current_point = smooth_point


        # ====================================================
        # RECORD
        # ====================================================

        if recording and current_point is not None:

            if len(trajectory) == 0:

                trajectory.append(
                    current_point
                )

            else:

                movement = distance(

                    trajectory[-1],

                    current_point

                )

                if movement > 2:

                    trajectory.append(
                        current_point
                    )


            if len(trajectory) > MAX_POINTS:

                trajectory.pop(0)


        # ====================================================
        # DRAW TRAJECTORY
        # ====================================================

        for i in range(
            1,
            len(trajectory)
        ):

            cv2.line(

                frame,

                trajectory[i - 1],

                trajectory[i],

                (255, 0, 255),

                3

            )


        # ====================================================
        # STATUS
        # ====================================================

        status = (

            "RECORDING"

            if recording

            else

            "READY"

        )


        cv2.putText(

            frame,

            status,

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 255, 0),

            2

        )


        cv2.putText(

            frame,

            f"Detected hands: {detected_hands}",

            (30, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Stable hands: {stable_hands}",

            (30, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Trajectory points: {len(trajectory)}",

            (30, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "R = Record",

            (30, height - 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "S = Stop",

            (30, height - 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "1 = HANDSHAKE",

            (30, height - 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "2 = NO HANDSHAKE",

            (300, height - 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.imshow(
            WINDOW,
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # START
        # ====================================================

        if key == ord("r"):

            trajectory = []

            smooth_point = None

            recording = True

            print()
            print("RECORDING")
            print("Perform the movement.")
            print("Press S when finished.")
            print()


        # ====================================================
        # STOP
        # ====================================================

        elif key == ord("s"):

            if recording:

                recording = False

                print()
                print("Recording stopped.")
                print(
                    f"Points: {len(trajectory)}"
                )

                print(
                    "Press 1 for HANDSHAKE"
                )

                print(
                    "Press 2 for NO HANDSHAKE"
                )


        # ====================================================
        # SAVE HANDSHAKE
        # ====================================================

        elif key == ord("1"):

            if (

                not recording

                and

                len(trajectory)
                >= MIN_POINTS

            ):

                trajectory_string = ";".join(

                    f"{x},{y}"

                    for x, y in trajectory

                )


                with open(

                    DATASET_FILE,

                    "a",

                    newline=""

                ) as file:

                    writer = csv.writer(
                        file
                    )

                    writer.writerow([

                        "HANDSHAKE",

                        trajectory_string

                    ])


                sample_number += 1


                print()
                print("HANDSHAKE SAMPLE SAVED")
                print(
                    f"Total samples: {sample_number}"
                )


                trajectory = []

                smooth_point = None


        # ====================================================
        # SAVE NO HANDSHAKE
        # ====================================================

        elif key == ord("2"):

            if (

                not recording

                and

                len(trajectory)
                >= MIN_POINTS

            ):

                trajectory_string = ";".join(

                    f"{x},{y}"

                    for x, y in trajectory

                )


                with open(

                    DATASET_FILE,

                    "a",

                    newline=""

                ) as file:

                    writer = csv.writer(
                        file
                    )

                    writer.writerow([

                        "NO_HANDSHAKE",

                        trajectory_string

                    ])


                sample_number += 1


                print()
                print("NO HANDSHAKE SAMPLE SAVED")
                print(
                    f"Total samples: {sample_number}"
                )


                trajectory = []

                smooth_point = None


        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            break


finally:

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print()
    print("Data collection stopped.")
    print(
        f"Samples collected: {sample_number}"
    )
    