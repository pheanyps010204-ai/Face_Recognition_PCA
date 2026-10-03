import cv2
import time
import numpy as np


# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Try to use a better webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# DRAW CORNER BRACKETS
# ============================================================

def draw_corner_box(
    frame,
    x,
    y,
    w,
    h,
    color=(0, 255, 0),
    thickness=2,
    corner_length=25
):

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
        (x + w, y),
        (x + w - corner_length, y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x + w, y),
        (x + w, y + corner_length),
        color,
        thickness
    )

    # Bottom-left
    cv2.line(
        frame,
        (x, y + h),
        (x + corner_length, y + h),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x, y + h),
        (x, y + h - corner_length),
        color,
        thickness
    )

    # Bottom-right
    cv2.line(
        frame,
        (x + w, y + h),
        (x + w - corner_length, y + h),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x + w, y + h),
        (x + w, y + h - corner_length),
        color,
        thickness
    )


# ============================================================
# DRAW TRANSPARENT PANEL
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.65
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# START
# ============================================================

print("Modern AI Face Detection started.")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    # ========================================================
    # TOP HEADER
    # ========================================================

    draw_panel(
        frame,
        0,
        0,
        frame.shape[1],
        75,
        alpha=0.75
    )

    cv2.putText(
        frame,
        "AI VISION",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "FACE DETECTION SYSTEM",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    status_x = frame.shape[1] - 270

    cv2.circle(
        frame,
        (status_x, 30),
        6,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        "SYSTEM ONLINE",
        (status_x + 15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


    # ========================================================
    # FACE BOXES
    # ========================================================

    for i, (x, y, w, h) in enumerate(faces):

        # Modern corner box
        draw_corner_box(
            frame,
            x,
            y,
            w,
            h,
            color=(0, 255, 255),
            thickness=2,
            corner_length=28
        )


        # ----------------------------------------------------
        # Face label panel
        # ----------------------------------------------------

        label_y = max(85, y - 45)

        draw_panel(
            frame,
            x,
            label_y,
            x + 180,
            label_y + 35,
            alpha=0.75
        )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (x + 10, label_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )


        # ----------------------------------------------------
        # Face number
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"ID {i + 1:02d}",
            (x + w - 55, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )


    # ========================================================
    # BOTTOM INFORMATION PANEL
    # ========================================================

    height, width = frame.shape[:2]

    draw_panel(
        frame,
        0,
        height - 65,
        width,
        height,
        alpha=0.8
    )


    # Face count

    cv2.putText(
        frame,
        f"FACES  {len(faces):02d}",
        (25, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # FPS

    cv2.putText(
        frame,
        f"FPS  {fps:.1f}",
        (180, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # Technology

    cv2.putText(
        frame,
        "OPENCV  |  HAAR CASCADE",
        (width - 330, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


    # ========================================================
    # SCANNING LINE
    # ========================================================

    scan_y = int(
        (time.time() * 120) % height
    )

    cv2.line(
        frame,
        (0, scan_y),
        (width, scan_y),
        (0, 180, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Vision - Face Detection",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Face detection stopped.")