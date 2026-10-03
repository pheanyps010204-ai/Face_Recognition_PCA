import cv2
import os
import time
import joblib
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(PROJECT_PATH, "models")


# ============================================================
# UNKNOWN PERSON THRESHOLD
# ============================================================

UNKNOWN_THRESHOLD = 19.28


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

scaler = joblib.load(
    os.path.join(MODEL_PATH, "scaler.pkl")
)

pca = joblib.load(
    os.path.join(MODEL_PATH, "pca_model.pkl")
)

svm = joblib.load(
    os.path.join(MODEL_PATH, "svm_pca_model.pkl")
)

print("Models loaded successfully.")
print("PCA components:", pca.n_components_)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Haar Cascade could not be loaded.")
    exit()


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break


    # --------------------------------------------------------
    # Mirror the webcam
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # ----------------------------------------------------
        # Extract face
        # ----------------------------------------------------

        face = gray[y:y+h, x:x+w]


        # ----------------------------------------------------
        # Resize to training size
        # ----------------------------------------------------

        face = cv2.resize(face, (64, 64))


        # ----------------------------------------------------
        # Flatten
        # 64 × 64 = 4096 features
        # ----------------------------------------------------

        face_flattened = face.reshape(1, -1)


        # ----------------------------------------------------
        # StandardScaler
        # ----------------------------------------------------

        face_scaled = scaler.transform(
            face_flattened
        )


        # ----------------------------------------------------
        # PCA
        # 4096 → 30 components
        # ----------------------------------------------------

        face_pca = pca.transform(
            face_scaled
        )


        # ----------------------------------------------------
        # SVM prediction
        # ----------------------------------------------------

        prediction = svm.predict(face_pca)[0]


        # ----------------------------------------------------
        # SVM decision score
        # ----------------------------------------------------

        scores = svm.decision_function(face_pca)

        max_score = float(
            np.max(scores)
        )


        # ====================================================
        # UNKNOWN PERSON REJECTION
        # ====================================================

        if max_score < UNKNOWN_THRESHOLD:

            name = "UNKNOWN"

        else:

            # IMPORTANT:
            # prediction is already "Person1", "Person2", etc.
            name = str(prediction)


        # ====================================================
        # FACE BOX
        # ====================================================

        color = (0, 255, 0)

        if name == "UNKNOWN":
            color = (0, 0, 255)


        # ----------------------------------------------------
        # Corner-style face box
        # ----------------------------------------------------

        corner_length = 30
        thickness = 3


        # Top-left
        cv2.line(
            frame,
            (x, y),
            (x + corner_length, y),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x, y),
            (x, y + corner_length),
            color,
            thickness
        )


        # Top-right
        cv2.line(
            frame,
            (x+w, y),
            (x+w-corner_length, y),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x+w, y),
            (x+w, y+corner_length),
            color,
            thickness
        )


        # Bottom-left
        cv2.line(
            frame,
            (x, y+h),
            (x+corner_length, y+h),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x, y+h),
            (x, y+h-corner_length),
            color,
            thickness
        )


        # Bottom-right
        cv2.line(
            frame,
            (x+w, y+h),
            (x+w-corner_length, y+h),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x+w, y+h),
            (x+w, y+h-corner_length),
            color,
            thickness
        )


        # ====================================================
        # NAME PANEL
        # ====================================================

        label_text = name

        cv2.rectangle(
            frame,
            (x, y-55),
            (x+260, y),
            (20, 20, 20),
            -1
        )


        # Name
        cv2.putText(
            frame,
            label_text,
            (x+15, y-30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SCORE
        # ====================================================

        score_text = f"Score: {max_score:.3f}"

        cv2.putText(
            frame,
            score_text,
            (x, y+h+30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # HEADER
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (1280, 90),
        (15, 15, 15),
        -1
    )


    cv2.putText(
        frame,
        "AI VISION / FACE RECOGNITION",
        (35, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # System status

    cv2.putText(
        frame,
        "SYSTEM ONLINE",
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        "PCA 30 | SVM",
        (1030, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (1100, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # FACE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"FACES DETECTED: {len(faces)}",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # PIPELINE INFORMATION
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 650),
        (1260, 710),
        (15, 15, 15),
        -1
    )


    cv2.putText(
        frame,
        "4096 FEATURES",
        (45, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "→",
        (230, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "30 PCA COMPONENTS",
        (275, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "→",
        (520, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "SVM",
        (565, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # UNKNOWN THRESHOLD
    # ========================================================

    threshold_text = (
        f"UNKNOWN THRESHOLD: {UNKNOWN_THRESHOLD:.1f}"
    )

    cv2.putText(
        frame,
        threshold_text,
        (850, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SCANNING LINE
    # ========================================================

    scan_y = int(
        150 +
        ((time.time() * 120) % 450)
    )

    cv2.line(
        frame,
        (40, scan_y),
        (1240, scan_y),
        (0, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Face Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Webcam closed.")