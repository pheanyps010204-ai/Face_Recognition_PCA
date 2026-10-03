import cv2
import os
import time


# ==========================================
# SETTINGS
# ==========================================

person_name = "Person1"

number_of_images = 10

dataset_path = os.path.join(
    "dataset",
    person_name
)


# ==========================================
# Create folder
# ==========================================

os.makedirs(
    dataset_path,
    exist_ok=True
)


# ==========================================
# DELETE OLD IMAGES
# ==========================================

old_files = [
    file
    for file in os.listdir(dataset_path)
    if file.lower().endswith(".jpg")
]

print(f"Found {len(old_files)} old images.")

for file in old_files:

    file_path = os.path.join(
        dataset_path,
        file
    )

    os.remove(file_path)


print("Old Person1 images deleted.")
print("Starting new collection...")
print()


# ==========================================
# LOAD FACE DETECTOR
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
# START CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


print("Camera started.")
print()
print("Look at the camera.")
print("Slowly change your face position.")
print("Press Q if you want to stop early.")
print()


# ==========================================
# COLLECT IMAGES
# ==========================================

count = 0

last_capture_time = 0

capture_interval = 0.5


while count < number_of_images:

    # --------------------------------------
    # Read camera frame
    # --------------------------------------

    ret, frame = camera.read()


    if not ret:

        print("ERROR: Could not read camera frame.")

        break


    # --------------------------------------
    # Convert to grayscale
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # --------------------------------------
    # Process detected face
    # --------------------------------------

    for (x, y, w, h) in faces:

        current_time = time.time()


        # ----------------------------------
        # Capture every 0.5 seconds
        # ----------------------------------

        if (
            current_time - last_capture_time
            >= capture_interval
        ):

            # Crop face

            face = gray[
                y:y+h,
                x:x+w
            ]


            # Resize to 64 × 64

            face = cv2.resize(
                face,
                (64, 64)
            )


            # ----------------------------------
            # File name
            # ----------------------------------

            filename = os.path.join(
                dataset_path,
                f"{count + 1:03d}.jpg"
            )


            # ----------------------------------
            # Save image
            # ----------------------------------

            cv2.imwrite(
                filename,
                face
            )


            count += 1

            last_capture_time = current_time


        # ----------------------------------
        # Draw face rectangle
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ----------------------------------
        # Show progress
        # ----------------------------------

        cv2.putText(
            frame,
            f"Images: {count}/{number_of_images}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Only use one face

        break


    # ======================================
    # Display webcam
    # ======================================

    cv2.imshow(
        "Collect Person1 Images",
        frame
    )


    # ======================================
    # Press Q to quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE CAMERA
# ==========================================

camera.release()

cv2.destroyAllWindows()


# ==========================================
# FINAL MESSAGE
# ==========================================

print()
print("==========================================")
print("Face collection finished.")
print("==========================================")
print(f"Person: {person_name}")
print(f"Images collected: {count}")
print(f"Saved in: {dataset_path}")
print("==========================================")