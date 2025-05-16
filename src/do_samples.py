import os
import time
import subprocess
import cv2

SAVE_DIR = "calib_imgs"
CAPTURE_DURATION = 60     
DELAY_BETWEEN = 0.5
TEMP_IMG = "/tmp/capture.jpg"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_frame(temp_path=TEMP_IMG):
    """Captures a single frame using rpicam-jpeg into the given path."""
    try:
        subprocess.run(
            ["rpicam-jpeg", "--output", temp_path, "--timeout", "100"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return cv2.imread(temp_path)
    except subprocess.CalledProcessError:
        print("Failed to capture image using rpicam-jpeg.")
        return None

def main():
    ensure_dir(SAVE_DIR)
    time.sleep(5)
    print("Start capturing images...")

    for i in range(CAPTURE_DURATION):
        frame = capture_frame()
        if frame is None:
            print(f"[{i+1}/60] Frame could not be captured, skipping.")
            continue

        filename = os.path.join(SAVE_DIR, f"calib_{i+1:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[{i+1}/60] Saved: {filename}")

        input()

    print("Capture complete.")

if __name__ == "__main__":
    main()
