import cv2
import numpy as np
import joblib
import os


# ==========================================
# 1. Find the project folder
# ==========================================

# This file is inside:
# Face_Recognition_PCA/src/webcam_recognition.py

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ==========================================
# 2. Load trained PCA and SVM models
# ==========================================

pca_path = os.path.join(
    project_path,
    "models",
    "pca_model.pkl"
)

svm_path = os.path.join(
    project_path,
    "models",
    "svm_pca_model.pkl"
)


# ==========================================
# 3. Check model files
# ==========================================

if not os.path.exists(pca_path):

    print("ERROR: PCA model not found.")
    print("Expected location:")
    print(pca_path)

    exit()


if not os.path.exists(svm_path):

    print("ERROR: SVM model not found.")
    print("Expected location:")
    print(svm_path)

    exit()


# ==========================================
# 4. Load models
# ==========================================

pca = joblib.load(pca_path)

svm = joblib.load(svm_path)

print("PCA model loaded successfully.")
print("SVM model loaded successfully.")


# ==========================================
# 5. Load Haar Cascade face detector
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    print("ERROR: Could not load face detector.")

    exit()


print("Face detector loaded successfully.")


# ==========================================
# 6. Open webcam
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not access camera.")

    exit()


print("Camera started.")
print("Look at the camera.")
print("Press Q to quit.")


# ==========================================
# 7. Unknown-person threshold
# ==========================================

UNKNOWN_THRESHOLD = -999


# ==========================================
# 8. Real-time face recognition
# ==========================================

try:

    while True:

        # ----------------------------------
        # Read webcam frame
        # ----------------------------------

        ret, frame = camera.read()


        if not ret:

            print("ERROR: Could not read camera frame.")

            break


        # ----------------------------------
        # Convert frame to grayscale
        # ----------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------
        # Detect faces
        # ----------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )


        # ----------------------------------
        # Process every detected face
        # ----------------------------------

        for (x, y, w, h) in faces:


            # ------------------------------
            # Crop face
            # ------------------------------

            face = gray[
                y:y+h,
                x:x+w
            ]


            # ------------------------------
            # Resize face
            # 64 × 64
            # ------------------------------

            face = cv2.resize(
                face,
                (64, 64)
            )


            # ------------------------------
            # Flatten image
            # 64 × 64 = 4096 features
            # ------------------------------

            face = face.reshape(
                1,
                -1
            )


            # ------------------------------
            # Apply PCA
            # 4096 → 10 components
            # ------------------------------

            face_pca = pca.transform(face)


            # ------------------------------
            # SVM decision scores
            # ------------------------------

            decision_scores = svm.decision_function(
                face_pca
            )


            # --------------------------------
            # Find predicted class
            # --------------------------------

            if decision_scores.ndim == 1:

                prediction_index = np.argmax(
                    decision_scores
                )

                confidence = decision_scores[
                    prediction_index
                ]

            else:

                prediction_index = np.argmax(
                    decision_scores[0]
                )

                confidence = decision_scores[
                    0,
                    prediction_index
                ]


            # --------------------------------
            # Get predicted person
            # --------------------------------

            predicted_person = svm.classes_[
                prediction_index
            ]


            # --------------------------------
            # Unknown-person decision
            # --------------------------------

            if confidence < UNKNOWN_THRESHOLD:

                prediction = "Unknown"

            else:

                prediction = predicted_person


            # --------------------------------
            # Draw face rectangle
            # --------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Display prediction
            # --------------------------------

            cv2.putText(
                frame,
                str(prediction),
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Display decision score
            # --------------------------------

            score_text = (
                f"Score: {confidence:.2f}"
            )


            cv2.putText(
                frame,
                score_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # ==================================
        # Display webcam window
        # ==================================

        cv2.imshow(
            "Real-Time Face Recognition",
            frame
        )


        # ==================================
        # Press Q to quit
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


finally:

    # ======================================
    # Close webcam
    # ======================================

    camera.release()

    cv2.destroyAllWindows()

    print("Webcam recognition stopped.")