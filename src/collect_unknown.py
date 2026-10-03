import cv2
import os
import time


# ============================================
# SETTINGS
# ============================================

person_name = "Unknown"
number_of_images = 10

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

save_path = os.path.join(
    project_path,
    "dataset",
    person_name
)

os.makedirs(
    save_path,
    exist_ok=True
)


# ============================================
# DELETE OLD UNKNOWN IMAGES
# ============================================

for filename in os.listdir(save_path):

    if filename.lower().endswith(".jpg"):

        os.remove(
            os.path.join(save_path, filename)
        )


# ============================================
# FACE DETECTOR
# ============================================

cascade_path = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)


# ============================================
# WEBCAM
# ============================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()


print("Unknown person image collection")
print("Press Q to quit.")


count = 0
last_capture = 0


# ============================================
# COLLECTION LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    for (x, y, w, h) in faces:

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face crop
        face = gray[
            y:y+h,
            x:x+w
        ]

        face = cv2.resize(
            face,
            (64, 64)
        )


        # Capture every 0.5 seconds
        current_time = time.time()

        if (
            current_time - last_capture >= 0.5
            and count < number_of_images
        ):

            count += 1

            filename = os.path.join(
                save_path,
                f"{count:03d}.jpg"
            )

            cv2.imwrite(
                filename,
                face
            )

            last_capture = current_time

            print(
                f"Saved image {count}/{number_of_images}"
            )


    # ========================================
    # DISPLAY STATUS
    # ========================================

    cv2.putText(
        frame,
        f"UNKNOWN TEST: {count}/{number_of_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Move your face slightly",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "Collect Unknown Face",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    if count >= number_of_images:
        break


# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()

print("\nCollection complete.")
print(
    f"{count} unknown images saved to:"
)
print(save_path)