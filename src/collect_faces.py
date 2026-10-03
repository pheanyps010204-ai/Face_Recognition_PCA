import cv2
import os

# -----------------------------
# Person name
# -----------------------------
person_name = input("Enter person's name: ")

# -----------------------------
# Create folder
# -----------------------------
save_path = os.path.join("dataset", person_name)

os.makedirs(save_path, exist_ok=True)

# -----------------------------
# Delete previous images
# -----------------------------
for file in os.listdir(save_path):
    if file.endswith(".jpg"):
        os.remove(os.path.join(save_path, file))

print("Previous images deleted.")

# -----------------------------
# Load face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Check detector
# -----------------------------
if face_detector.empty():
    print("Could not load face detector.")
    exit()

# -----------------------------
# Open webcam
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access camera.")
    exit()

count = 0

print("Starting camera...")
print("Look at the camera.")
print("Press Q to stop.")

# -----------------------------
# Start collecting images
# -----------------------------
while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------
    # Detect faces
    # -----------------------------
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # -----------------------------
    # Process detected faces
    # -----------------------------
    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # -----------------------------
        # Crop face
        # -----------------------------
        face = gray[y:y+h, x:x+w]

        # -----------------------------
        # Resize face to 64 × 64
        # -----------------------------
        face = cv2.resize(
            face,
            (64, 64)
        )

        # -----------------------------
        # Save image
        # -----------------------------
        filename = os.path.join(
            save_path,
            f"{count}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        count += 1

        # Stop after 100 images
        if count >= 100:
            break

    # -----------------------------
    # Display image count
    # -----------------------------
    cv2.putText(
        frame,
        f"Images: {count}/100",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Show camera
    # -----------------------------
    cv2.imshow(
        "Face Collection",
        frame
    )

    # -----------------------------
    # Stop after 100 images
    # -----------------------------
    if count >= 100:
        break

    # -----------------------------
    # Press Q to stop
    # -----------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Release camera
# -----------------------------
camera.release()

# Close windows
cv2.destroyAllWindows()

# -----------------------------
# Final message
# -----------------------------
print()
print(f"Collected {count} images.")
print(f"Saved in: {save_path}")