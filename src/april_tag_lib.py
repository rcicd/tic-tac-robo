import os
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import subprocess

CAMERA_MATRIX: np.ndarray = np.array(
    [
        [2.76886766e+03, 0.0, 1.64025593e+03],
        [0.0, 3.57857333e+03, 1.23645626e+03],
        [0.0, 0.0, 1.0],
    ]
)
DIST_COEFFS: np.ndarray = np.array(
    [
        -0.4707324,
        -0.08520538,
        0.00724763,
        -0.00829328,
        0.45304851,
    ]
)

def capture_image(path="/tmp/frame.jpg"):
    subprocess.run(["rpicam-jpeg", "--output", path, "--timeout", "100",
                    ], check=True)
    img = cv2.imread(path)
    return img

def capture_frame(cam_index: int = 0) -> np.ndarray:
    return capture_image()

def _rotation_to_euler(rvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return float(x), float(y), float(z)

class ArucoTracker:
    def __init__(
        self,
        camera_matrix: np.ndarray = CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DIST_COEFFS,
        marker_size: float = 0.027,
        dictionary: int = aruco.DICT_APRILTAG_36h11,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size
        self._dict = aruco.getPredefinedDictionary(dictionary)
        self._params = aruco.DetectorParameters()

    def detect_markers(
        self,
        frame: np.ndarray,
        *,
        draw_axes: bool = False,
        axes_length: float = 0.1,
    ):
        frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self._dict, parameters=self._params)
        results = []
        if ids is None:
            return results
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
        )
        for idx, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rvecs[idx], tvecs[idx]
            if draw_axes:
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, axes_length)
            orientation = _rotation_to_euler(rvec)
            distance = float(np.linalg.norm(tvec))
            results.append(
                (
                    int(marker_id),
                    {
                        "rvec": rvec,
                        "tvec": tvec,
                        "orientation": orientation,
                        "distance": distance,
                    },
                )
            )
        return results

    @staticmethod
    def relative_pose(marker_a, marker_b):
        rvec1, tvec1 = marker_a["rvec"].reshape(3), marker_a["tvec"].reshape(3)
        rvec2, tvec2 = marker_b["rvec"].reshape(3), marker_b["tvec"].reshape(3)
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R_rel = R1.T @ R2
        t_rel = R1.T @ (tvec2 - tvec1)
        r_rel, _ = cv2.Rodrigues(R_rel)

        yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        return {
            "x": t_rel[0],
            "y": t_rel[1],
            "z": t_rel[2],
            "yaw_deg": np.degrees(yaw_rad)
        }


def markers_relative_to_home(
    frame: np.ndarray,
    tracker: ArucoTracker,
    *,
    home_id: int = 3,
):
    detections = tracker.detect_markers(frame, draw_axes=False)
    if not detections:
        raise ValueError("No markers detected in the frame.")
    try:
        home_pose = next(data for mid, data in detections if mid == home_id)
    except StopIteration:
        raise ValueError(f"Home marker id={home_id} not found.")
    rel = []
    for mid, data in detections:
        if mid == home_id:
            continue
        rel.append((mid, ArucoTracker.relative_pose(home_pose, data)))
    return rel

def rvec_to_z_angle(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

def find_frame_with_home_marker(
    tracker: ArucoTracker,
    home_id: int = 3,
    max_tries: int = 20,
    cam_index: int = 0,
    save_dir: str = None,
):
    last_frame = None
    last_error = None

    for attempt in range(max_tries):
        frame = capture_frame(cam_index)
        last_frame = frame
        try:
            rel = markers_relative_to_home(frame, tracker, home_id=home_id)
            print(f"[INFO] Found home marker (id={home_id}) on attempt {attempt+1}")
            annotated = frame.copy()
            detections = tracker.detect_markers(annotated, draw_axes=True)
            ids = [mid for mid, _ in detections]
            print("[INFO] Marker IDs in frame:", ids)
            save_path = "tags_found.jpg" if save_dir is None else os.path.join(save_dir, "tags_found.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"[INFO] Annotated frame saved -> {save_path}")
            for mid, pose in rel:
                print(f"[INFO] Marker {mid} relative to home: "
                    f"x={pose['x']:.3f} m, y={pose['y']:.3f} m, z={pose['z']:.3f} m, "
                    f"yaw={pose['yaw_deg']:.1f}°")
            return rel, annotated
        except ValueError as err:
            last_error = err
            print(f"[WARN] Attempt {attempt+1}: {err}")
        time.sleep(0.05)

    fail_path = "tags_fail.jpg" if save_dir is None else os.path.join(save_dir, "tags_fail.jpg")
    cv2.imwrite(fail_path, last_frame)
    print(f"[ERROR] Home marker (id={home_id}) not found in {max_tries} tries! Last frame saved → {fail_path}")
    if last_error is not None:
        raise RuntimeError(str(last_error))
    else:
        raise RuntimeError("Unknown error during marker search.")

if __name__ == "__main__":
    tracker = ArucoTracker()
    try:
        rel, annotated = find_frame_with_home_marker(tracker, home_id=3, max_tries=20)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
