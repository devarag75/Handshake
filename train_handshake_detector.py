import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "handshake_detection_dataset.csv"

MODEL_FILE = "handshake_detector.pkl"

POINTS = 40


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATASET_FILE
)

print()
print("==============================================")
print("HANDSHAKE AI TRAINING")
print("==============================================")
print()

print("Dataset samples:", len(df))

print()
print("Class distribution:")
print(
    df["label"].value_counts()
)


# ============================================================
# PARSE TRAJECTORY
# ============================================================

def parse_trajectory(text):

    points = []

    try:

        for item in str(text).split(";"):

            x, y = item.split(",")

            points.append([
                float(x),
                float(y)
            ])

    except:

        return None


    if len(points) < 5:

        return None


    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# RESAMPLE
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    trajectory
):

    trajectory = resample_trajectory(
        trajectory,
        POINTS
    )


    # --------------------------------------------------------
    # Normalize starting position
    # --------------------------------------------------------

    trajectory = (
        trajectory
        - trajectory[0]
    )


    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = np.diff(
        trajectory,
        axis=0
    )


    speed = np.linalg.norm(
        velocity,
        axis=1
    )


    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    angles = np.arctan2(

        velocity[:, 1],

        velocity[:, 0]

    )


    angle_difference = np.diff(
        angles
    )


    angle_difference = (
        (angle_difference + np.pi)
        % (2 * np.pi)
    ) - np.pi


    direction_changes = np.sum(

        np.abs(
            angle_difference
        ) > 0.5

    )


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    acceleration = np.diff(
        speed
    )


    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    path_length = np.sum(
        speed
    )


    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    displacement = np.linalg.norm(

        trajectory[-1]
        -
        trajectory[0]

    )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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
# BUILD DATASET
# ============================================================

X = []

y = []


for _, row in df.iterrows():

    trajectory = parse_trajectory(
        row["trajectory"]
    )


    if trajectory is None:

        continue


    features = extract_features(
        trajectory
    )


    X.append(features)

    y.append(
        row["label"]
    )


X = np.array(X)

y = np.array(y)


print()
print("Usable samples:", len(X))

print(
    "Feature count:",
    X.shape[1]
)


# ============================================================
# CHECK CLASSES
# ============================================================

classes, counts = np.unique(
    y,
    return_counts=True
)


print()
print("Classes:")

for c, count in zip(
    classes,
    counts
):

    print(
        f"{c}: {count}"
    )


if len(classes) < 2:

    print()
    print(
        "ERROR: Need both HANDSHAKE and NO_HANDSHAKE samples."
    )

    raise SystemExit


if np.min(counts) < 10:

    print()
    print(
        "ERROR: Every class should have at least 10 samples."
    )

    raise SystemExit


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_split=3,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


print()
print("Training AI model...")
print()


model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(

    y_test,

    predictions

)


print()
print("==============================================")
print("MODEL RESULTS")
print("==============================================")
print()

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y_test,
        predictions
    )
)

print()

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_data = {

    "model": model,

    "points": POINTS

}


with open(

    MODEL_FILE,

    "wb"

) as file:

    pickle.dump(
        model_data,
        file
    )


print()
print("==============================================")
print("MODEL SAVED")
print("==============================================")
print()
print(
    MODEL_FILE
)
print()