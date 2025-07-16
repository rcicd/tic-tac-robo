from __future__ import annotations

import sys
import random
import queue
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from arm_lib import (
    DeviceConnection,
    BaseClient,
    BaseCyclicClient,
    move_to_home,
    set_gripper,
    cartesian_action_movement,
)
from aprli_tag_new import ArucoTracker, capture_image, get_session_logger
from tic_tac_toe import find_best_move

# ────────────────────────────────
# Constants from main.py
# ────────────────────────────────
BOARD = 3
CELL = 0.08
FIELD_SIZE = CELL * BOARD
MARKER_SIZE = 0.04
TILE_SIZE = 0.05
HOME_SHIFT = 0.024

# ────────────────────────────────
# Helper function for marker annotation
# ────────────────────────────────

def annotate_markers_with_symbols(image, target_marker_id=None):
    print(f"[DEBUG] annotate_markers_with_symbols called with image shape: {image.shape}, target_marker_id: {target_marker_id}")
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        print(f"[DEBUG] Applied preprocessing: grayscale + histogram equalization")
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.1
        
        print(f"[DEBUG] Created ArUco dictionary (DICT_APRILTAG_36h11) and parameters with corner refinement")
        
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            print(f"[DEBUG] Used new ArUco API")
        except AttributeError:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            print(f"[DEBUG] Used old ArUco API")
        
        print(f"[DEBUG] ArUco detection result: marker_ids={marker_ids}, corners_count={len(marker_corners) if marker_corners else 0}")
        
        if marker_ids is not None and len(marker_ids) > 0:
            print(f"Found {len(marker_ids)} markers: {marker_ids.flatten()}")
            
            for i, marker_id in enumerate(marker_ids.flatten()):
                corners = marker_corners[i][0]
                
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                
                min_x = int(np.min(corners[:, 0]))
                max_x = int(np.max(corners[:, 0]))
                min_y = int(np.min(corners[:, 1]))
                max_y = int(np.max(corners[:, 1]))
                
                marker_width = max_x - min_x
                marker_height = max_y - min_y
                marker_size = int(min(marker_width, marker_height))
                
                thickness = max(8, marker_size // 6)
                
                print(f"Drawing on marker {marker_id} at ({center_x}, {center_y}), size: {marker_width}x{marker_height}")
                
                if marker_id % 2 == 1:
                    color = (0, 0, 255)
                    
                    p0, p1, p2, p3 = corners.astype(int)
                    cv2.line(image, p0, p2, color, thickness)
                    cv2.line(image, p1, p3, color, thickness)

                    print(f"Drew red X on marker {marker_id}")
                elif marker_id > 0 and marker_id % 2 == 0:
                    color = (255, 0, 0)
                    rect = cv2.minAreaRect(corners)
                    (cx, cy), (w, h), angle = rect
                    cv2.ellipse(image, (int(cx), int(cy)),
                                (int(w/2), int(h/2)),
                                angle, 0, 360, color, thickness)

                    print(f"Drew blue O on marker {marker_id}")
                    
                border_thickness = 2
                border_color = (0, 255, 0)
                
                if target_marker_id is not None and marker_id == target_marker_id:
                    border_thickness = 12
                    border_color = (0, 255, 0)
                    print(f"Drawing thick green border for target marker {marker_id}")
                
                cv2.polylines(image, [corners.astype(int)], True, border_color, border_thickness)
        else:
            print("No ArUco markers detected in image")
            
    except Exception as e:
        print(f"Error in annotate_markers_with_symbols: {e}")
        import traceback
        traceback.print_exc()
    
    return image


# ────────────────────────────────
# Helper functions from main.py
# ────────────────────────────────


def pose_to_cell(pose):
    print(pose)
    x = -(pose["x"] + HOME_SHIFT)
    y = -(pose["y"] + HOME_SHIFT)

    if x > FIELD_SIZE or x < 0 or y > FIELD_SIZE or y < 0:
        print(x, y)
        print(f"[INFO] Marker outside the field: x={x:.3f}, y={y:.3f}")
        return None

    col = int(x // CELL)
    row = int(y // CELL)

    if 0 <= col < BOARD and 0 <= row < BOARD:
        return row, col
    return None


def get_marker_type(marker_id):
    if marker_id % 2 == 1:
        return "X"
    elif marker_id > 0 and marker_id % 2 == 0:
        return "O"
    return None


def base_position(base, base_cyclic, dx=0, dy=0, dtz=0):
    cartesian_action_movement(
        base, base_cyclic, x=-0.05 + dx, y=-0.15 + dy, z=-0.4, tx=90, tz=dtz
    )


def find_marker_by_type(
    tracker: ArucoTracker,
    frame,
    *,
    marker_type: str,  # 'X' or 'O'
    home_id: int = 0,
):
    try:
        detections = tracker.detect_markers(frame)
        if not detections:
            raise ValueError("No markers detected in the frame.")

        try:
            home_pose = next(data for mid, data in detections if mid == home_id)
        except StopIteration:
            raise ValueError(f"Home marker id={home_id} not visible.")
        
        for mid, data in detections:
            if get_marker_type(mid) != marker_type:
                continue

            rel_pose = ArucoTracker.relative_pose(home_pose, data)
            x = rel_pose["x"]
            y = rel_pose["y"]
            z = rel_pose["z"]
            yaw = rel_pose["yaw_deg"]

            if x > -FIELD_SIZE and x < 0 and y > -FIELD_SIZE and y < 0:
                continue

            print(
                f"[INFO] marker {mid} ({marker_type}) is found "
                f"x = {x:.3f} "
                f"y = {y:.3f} "
                f"z = {z:.3f} "
                f"z_angle={yaw:.1f}°"
            )
            return mid, x, y, z, yaw

        raise ValueError(f"Marker type {marker_type} is not visible in the frame.")

    except ValueError as err:
        raise RuntimeError(f"Marker type {marker_type} not found! Error: {err}")


def read_board(frame, tracker, home_id=0):
    from aprli_tag_new import markers_relative_to_home
    logger = get_session_logger()
    logger.logger.info(f"Reading board state with home marker id={home_id}")

    logger.save_image(frame, "board_reading_input.jpg")

    rel = markers_relative_to_home(frame, tracker, home_id=home_id)
    board = [["." for _ in range(BOARD)] for _ in range(BOARD)]

    logger.logger.info(f"Found {len(rel)} markers relative to home")
    detected_markers = []

    for mid, pose in rel:
        logger.logger.debug(f"Processing marker {mid}: pose={pose}")
        cell = pose_to_cell(pose)
        if cell is None:
            logger.logger.debug(f"Marker {mid} is outside the board area")
            continue

        r, c = cell
        marker_type = get_marker_type(mid)
        if marker_type:
            board[r][c] = marker_type
            detected_markers.append(
                {
                    "marker_id": mid,
                    "marker_type": marker_type,
                    "position": {"row": r, "col": c},
                    "pose": pose,
                }
            )
            logger.logger.debug(
                f"Placed marker {mid} ({marker_type}) at position ({r}, {c})"
            )
        else:
            logger.logger.debug(f"Marker {mid} has unknown type")

    return board


# ────────────────────────────────
#  Model
# ────────────────────────────────


class TicTacToeModel(QObject):
    board_changed = Signal()
    game_over = Signal(str)  # 'X', 'O' or 'Draw'

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.board: List[List[str]] = [["" for _ in range(3)] for _ in range(3)]
        self.board_changed.emit()

    # helpers ---------------------------------------------------------

    def _lines(self):
        b = self.board
        for i in range(3):
            yield b[i]
            yield [b[0][i], b[1][i], b[2][i]]
        yield [b[0][0], b[1][1], b[2][2]]
        yield [b[0][2], b[1][1], b[2][0]]

    def check_winner(self):
        for line in self._lines():
            if line[0] and all(c == line[0] for c in line):
                return line[0]
        if all(cell for row in self.board for cell in row):
            return "Draw"
        return None

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if not self.board[r][c]]

    def place(self, s: str, r: int, c: int):
        if self.board[r][c]:
            raise ValueError("Cell busy")
        self.board[r][c] = s
        self.board_changed.emit()
        w = self.check_winner()
        if w:
            self.game_over.emit(w)


# ────────────────────────────────
#  Worker threads
# ────────────────────────────────


class CameraThread(QThread):
    frame_ready = Signal(QImage)

    def __init__(self, fps: int = 10):
        super().__init__()
        self._running = True
        self._interval = 1.0 / fps
        self._last_cv_frame = None
        self._target_marker_id = None

    def run(self):
        while self._running:
            try:
                cv_img = capture_image()
                self._last_cv_frame = cv_img
                
                # Annotate markers with symbols for display (X and O)
                annotated_img = annotate_markers_with_symbols(cv_img.copy(), self._target_marker_id)
                
                # Convert annotated image to QImage for display
                rgb_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
            except Exception as e:
                print(f"CameraThread error: {e}")
                # Create a black image with error text
                img = QImage(640, 480, QImage.Format_RGB888)
                img.fill(Qt.black)
                painter = QPainter(img)
                painter.setPen(Qt.white)
                painter.setFont(QFont("Sans", 12))
                painter.drawText(img.rect(), Qt.AlignCenter, f"Camera Error: {e}")
                painter.end()
                self.frame_ready.emit(img)
            time.sleep(self._interval)

    def get_last_frame(self):
        import copy
        return copy.deepcopy(self._last_cv_frame) if self._last_cv_frame is not None else None

    def set_target_marker(self, marker_id: Optional[int]):
        self._target_marker_id = marker_id

    def stop(self):
        self._running = False
        self.wait()


class RobotThread(QThread):
    predicted_move = Signal(int, int)
    move_done = Signal(int, int, str)
    calibration_done = Signal()
    target_marker_changed = Signal(int)

    def __init__(
        self, model: TicTacToeModel, base: BaseClient, base_cyclic: BaseCyclicClient, camera_thread: CameraThread
    ):
        super().__init__()
        self.model = model
        self._q: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
        self._running = True
        self.base = base
        self.base_cyclic = base_cyclic
        self.camera_thread = camera_thread
        self.tracker = ArucoTracker(marker_size=MARKER_SIZE, ema_alpha=1.0)

    # API -------------------------------------------------------------

    def calibrate(self):
        self._q.put(("calibrate", None))

    def make_move(self, symbol: str):
        self._q.put(("move", symbol))

    def stop(self):
        self._q.put(("stop", None))
        self.wait()

    # loop ------------------------------------------------------------

    def run(self):
        while self._running:
            cmd, arg = self._q.get()
            if cmd == "stop":
                self._running = False
            elif cmd == "calibrate":
                self._do_calibration()
            elif cmd == "move":
                self._do_move(arg)  # type: ignore[arg‑type]

    # internals -------------------------------------------------------

    def _do_calibration(self):
        move_to_home(self.base)
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)
        set_gripper(self.base, 80)
        self.calibration_done.emit()

    def _do_move(self, symbol: str):
        # 0. Get current frame from camera
        img = self.camera_thread.get_last_frame()
        if img is None:
            print("[ERROR] No frame available from camera")
            return
    
        # 1. Find robot's marker to pick up.
        try:
            marker_id, x, y, _, yaw = find_marker_by_type(
                self.tracker, img, marker_type=symbol
            )
            print(f"Marker is found {marker_id} of type {symbol}")
            
            self.target_marker_changed.emit(marker_id)

            detections = self.tracker.detect_markers(img)
            home_yaw = None
            for mid, data in detections:
                if mid == 0:  # home marker id
                    home_yaw = data["orientation"][2]
                    print(f"Home marker yaw: {home_yaw:.1f}°")
                    break

        except RuntimeError as e:
            print(f"[ERROR] {e}")
            self.target_marker_changed.emit(-1)
            return

        # 2. Read board state
        board_from_cam = read_board(img, self.tracker)
        board_from_cam = [[c.replace(".", "") for c in row] for row in board_from_cam]

        # Sync model with camera state
        self.model.board = board_from_cam
        self.model.board_changed.emit()
        time.sleep(0.1)  # Allow UI to update

        # 3. Find best move
        move = find_best_move(board_from_cam, symbol)

        if move is None:
            print("No valid move found for the robot.")
            self.target_marker_changed.emit(-1)
            winner = self.model.check_winner()
            if winner:
                self.model.game_over.emit(winner)
            return

        # 4. Execute move
        r, c = move
        self.predicted_move.emit(r, c)

        local_motion = np.array([x, y])
        print(local_motion)
        print(np.rad2deg(home_yaw))
        adj_home_yaw = home_yaw
        adj_home_yaw += np.pi / 2  # Adjust home yaw to match robot's orientation
        adj_home_yaw *= -1
        rotation_matrix = np.array([
            [np.cos(adj_home_yaw), -np.sin(adj_home_yaw)],
            [np.sin(adj_home_yaw),  np.cos(adj_home_yaw)],
        ])

        motion = rotation_matrix @ local_motion
        print(motion)  
        # print(yaw)
        # exit()

        global_yaw = yaw + np.rad2deg(home_yaw)
        while global_yaw > 180:
            global_yaw -= 360
        while global_yaw <= -180:
            global_yaw += 360

        # Pick and place sequence
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)
        cartesian_action_movement(self.base, self.base_cyclic, x=motion[0], y=motion[1])
        cartesian_action_movement(self.base, self.base_cyclic, tz=global_yaw)

        set_gripper(self.base, 30)
        cartesian_action_movement(self.base, self.base_cyclic, z=-0.05)
        set_gripper(self.base, 70)
        cartesian_action_movement(self.base, self.base_cyclic, z=0.1)

        move_to_home(self.base)
        base_position(self.base, self.base_cyclic, dy=0.18, dx=0.03)
        cartesian_action_movement(self.base, self.base_cyclic, tz=25)

        target_x = -(HOME_SHIFT + CELL * c + CELL / 2)
        target_y = -(HOME_SHIFT + CELL * r + CELL / 2) + 0.02
        local_motion = [target_x, target_y]
        motion = rotation_matrix @ local_motion
        cartesian_action_movement(self.base, self.base_cyclic, x=motion[0])
        cartesian_action_movement(self.base, self.base_cyclic, y=motion[1])
        cartesian_action_movement(self.base, self.base_cyclic, tz = np.rad2deg(-home_yaw))

        cartesian_action_movement(self.base, self.base_cyclic, z=-0.045)
        set_gripper(self.base, 50)
        move_to_home(self.base)

        # 5. Update model
        try:
            self.model.place(symbol, r, c)
            self.move_done.emit(r, c, symbol)
            self.target_marker_changed.emit(-1)
        except ValueError as e:
            print(f"Error placing marker in model: {e}")
            self.target_marker_changed.emit(-1)


# ────────────────────────────────
#  Views
# ────────────────────────────────


class MainMenuView(QWidget):
    play_clicked = Signal()
    cam_clicked = Signal()
    cfg_clicked = Signal()

    def __init__(self):
        super().__init__()
        box = QVBoxLayout(self)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(50)
        self.play_btn.clicked.connect(self.play_clicked)

        self.cam_btn = QPushButton("Check camera")
        self.cam_btn.setFixedHeight(50)
        self.cam_btn.clicked.connect(self.cam_clicked)

        self.cfg_btn = QPushButton("Configure")
        self.cfg_btn.setFixedHeight(50)
        self.cfg_btn.clicked.connect(self.cfg_clicked)

        box.addWidget(self.play_btn)
        box.addWidget(self.cam_btn)
        box.addWidget(self.cfg_btn)
        box.addStretch(1)

    def set_play_enabled(self, state: bool):
        self.play_btn.setEnabled(state)


class CameraView(QWidget):
    back_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.img = QLabel(alignment=Qt.AlignCenter)
        self.img.setMinimumSize(640, 480)
        back = QPushButton("Back")
        back.clicked.connect(self.back_clicked)
        lay = QVBoxLayout(self)
        lay.addWidget(self.img)
        lay.addWidget(back)

    def update_frame(self, frame: QImage):
        self.img.setPixmap(QPixmap.fromImage(frame.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))


class GameView(QWidget):
    surrender_clicked = Signal()
    human_move_done = Signal()

    def __init__(self):
        super().__init__()
        self.frame = QLabel(alignment=Qt.AlignCenter)
        self.frame.setMinimumSize(400, 300)
        self.frame.setStyleSheet("border: 1px solid gray;")

        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)

        self.cells: List[List[QLabel]] = [[QLabel() for _ in range(3)] for _ in range(3)]
        grid = QGridLayout()
        grid.setSpacing(2)
        for r in range(3):
            for c in range(3):
                lbl = self.cells[r][c]
                lbl.setFixedSize(80, 80)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("""
                    font-size: 32pt;
                    border: 2px solid #555;
                    background-color: #1e1e1e;
                    color: white;
                    border-radius: 5px;
                """)
                grid.addWidget(lbl, r, c)
        
        btn = QPushButton("Give up")
        btn.setFixedHeight(40)
        btn.clicked.connect(self.surrender_clicked)
        
        self.done_btn = QPushButton("My turn is done")
        self.done_btn.setFixedHeight(40)
        self.done_btn.clicked.connect(self.human_move_done)

        right_layout = QVBoxLayout(right_panel)
        right_layout.addStretch(1)
        right_layout.addLayout(grid)
        right_layout.addStretch(1)
        right_layout.addWidget(btn)
        right_layout.addWidget(self.done_btn)
        right_layout.addStretch(1)

        main = QGridLayout(self)
        main.setContentsMargins(5, 5, 5, 5)
        main.addWidget(self.frame, 0, 0)
        main.addWidget(right_panel, 0, 1)
        main.setColumnStretch(0, 1)
        main.setColumnStretch(1, 0)

    def update_board(
        self, board: List[List[str]], hint: Optional[Tuple[int, int]] = None
    ):
        for r in range(3):
            for c in range(3):
                lbl = self.cells[r][c]
                s = board[r][c]
                if s == "X":
                    lbl.setText("X")
                    lbl.setStyleSheet("""
                        font-size: 32pt;
                        color: #ff4444;
                        border: 2px solid #555;
                        background-color: #1e1e1e;
                        border-radius: 5px;
                    """)
                elif s == "O":
                    lbl.setText("O")
                    lbl.setStyleSheet("""
                        font-size: 32pt;
                        color: #4488ff;
                        border: 2px solid #555;
                        background-color: #1e1e1e;
                        border-radius: 5px;
                    """)
                else:
                    lbl.setText("")
                    lbl.setStyleSheet("""
                        font-size: 32pt;
                        border: 2px solid #555;
                        background-color: #1e1e1e;
                        color: white;
                        border-radius: 5px;
                    """)
        if hint and not board[hint[0]][hint[1]]:
            r, c = hint
            lbl = self.cells[r][c]
            lbl.setText("•")
            lbl.setStyleSheet("""
                font-size: 32pt;
                color: #888;
                border: 2px solid #777;
                background-color: #2a2a2a;
                border-radius: 5px;
            """)

    def update_frame(self, frame: QImage):
        scaled_pixmap = QPixmap.fromImage(
            frame.scaled(
                self.frame.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        )
        self.frame.setPixmap(scaled_pixmap)


# ────────────────────────────────
#  Controller
# ────────────────────────────────


class AppController(QObject):
    def __init__(self, stack: QStackedWidget):
        super().__init__()
        self.stack = stack
        self.model = TicTacToeModel()

        # views
        self.menu = MainMenuView()
        self.camera = CameraView()
        self.game = GameView()
        for w in (self.menu, self.camera, self.game):
            self.stack.addWidget(w)

        # Robot Connection
        self.device_connection = None
        self.router = None
        self.robot_connected = False
        try:
            self.device_connection = DeviceConnection.createTcpConnection()
            self.router = self.device_connection.__enter__()  # Initialize the connection
            base = BaseClient(self.router)
            base_cyclic = BaseCyclicClient(self.router)
            self.robot_connected = True
        except Exception as e:
            QMessageBox.critical(None, "Connection Error", f"Failed to connect to robot: {e}")
            base, base_cyclic = None, None

        # threads
        self.cam_thread = CameraThread()
        if self.robot_connected:
            self.robot_thread = RobotThread(self.model, base, base_cyclic, self.cam_thread)

        self.cam_thread.start()
        if self.robot_connected:
            self.robot_thread.start()

        # connect threads → views
        self.cam_thread.frame_ready.connect(self.camera.update_frame)
        self.cam_thread.frame_ready.connect(self.game.update_frame)
        if self.robot_connected:
            self.robot_thread.predicted_move.connect(self._on_hint)
            self.robot_thread.calibration_done.connect(self._calibration_done)
            self.robot_thread.move_done.connect(self._on_robot_move_done)
            self.robot_thread.target_marker_changed.connect(self._on_target_marker_changed)

        # model → views
        self.model.board_changed.connect(self._refresh_board)
        self.model.game_over.connect(self._game_over)

        # menu buttons
        self.menu.play_clicked.connect(self.start_game)
        self.menu.cam_clicked.connect(lambda: self.stack.setCurrentWidget(self.camera))
        self.menu.cfg_clicked.connect(self.run_calibration)
        self.camera.back_clicked.connect(lambda: self.stack.setCurrentWidget(self.menu))
        self.game.surrender_clicked.connect(self.end_game)
        self.game.human_move_done.connect(self.trigger_robot_move)

        # state
        self._hint_pos: Optional[Tuple[int, int]] = None
        self._target_marker_id: Optional[int] = None
        self.human_symbol = "X"
        self.robot_symbol = "O"

        if not self.robot_connected:
            self.menu.set_play_enabled(False)
            self.menu.cfg_btn.setEnabled(False)

    # ─── Calibration -------------------------------------------------

    def run_calibration(self):
        if not self.robot_connected: return
        self.menu.set_play_enabled(False)
        self.robot_thread.calibrate()
        QMessageBox.information(None, "Calibrate", "Robot is moving to grab pose…")

    def _calibration_done(self):
        QMessageBox.information(None, "Calibrate", "Place the board and press OK.")
        move_to_home(self.robot_thread.base)
        self.menu.set_play_enabled(True)
        self.stack.setCurrentWidget(self.menu)

    # ─── Gameplay ----------------------------------------------------

    def _ask_player_symbol(self) -> str:
        dlg = QMessageBox()
        dlg.setWindowTitle("Choose your symbol")
        dlg.setText("Play as:")
        x_btn = dlg.addButton("X", QMessageBox.AcceptRole)
        o_btn = dlg.addButton("O", QMessageBox.AcceptRole)
        dlg.addButton("Random", QMessageBox.AcceptRole)
        dlg.exec()
        clicked = dlg.clickedButton()
        if clicked == x_btn:
            return "X"
        if clicked == o_btn:
            return "O"
        return random.choice(["X", "O"])

    def start_game(self):
        if not self.robot_connected: return
        self.human_symbol = self._ask_player_symbol()
        self.robot_symbol = "O" if self.human_symbol == "X" else "X"

        self.model.reset()
        self._hint_pos = None
        self._refresh_board()
        self.stack.setCurrentWidget(self.game)

        if self.robot_symbol == "X":
            self.game.done_btn.setEnabled(False)
            self.robot_thread.make_move(self.robot_symbol)
        else:
            self.game.done_btn.setEnabled(True)
            QMessageBox.information(None, "Your turn", "Place your marker and press 'My turn is done'")


    def trigger_robot_move(self):
        self.game.done_btn.setEnabled(False)
        self.robot_thread.make_move(self.robot_symbol)

    def _on_robot_move_done(self, r: int, c: int, symbol: str):
        if not self.model.check_winner():
            self.game.done_btn.setEnabled(True)
            QMessageBox.information(None, "Your turn", "Place your marker and press 'My turn is done'")


    def end_game(self):
        self.model.reset()
        self._target_marker_id = None
        self.cam_thread.set_target_marker(None)
        self.stack.setCurrentWidget(self.menu)

    # ─── Helpers -----------------------------------------------------

    def _refresh_board(self):
        self.game.update_board(self.model.board, self._hint_pos)

    def _on_hint(self, r: int, c: int):
        self._hint_pos = (r, c)
        self._refresh_board()

    def _on_target_marker_changed(self, marker_id: int):

        if marker_id == -1:
            self._target_marker_id = None
        else:
            self._target_marker_id = marker_id
        
        self.cam_thread.set_target_marker(self._target_marker_id)
        print(f"[DEBUG] Target marker set to: {self._target_marker_id}")

    def _game_over(self, winner: str):
        msg = "Draw!" if winner == "Draw" else f"Winner: {winner}"
        QMessageBox.information(None, "Game over", msg)
        self.end_game()

    # ─── Clean‑up ----------------------------------------------------

    def shutdown(self):
        self.cam_thread.stop()
        if self.robot_connected:
            self.robot_thread.stop()
        if self.device_connection:
            self.device_connection.__exit__(None, None, None)


# ────────────────────────────────
#  Entry‑point
# ────────────────────────────────


def main() -> None:  # noqa: D401
    app = QApplication(sys.argv)
    stack = QStackedWidget()
    controller = AppController(stack)

    app.aboutToQuit.connect(controller.shutdown)

    stack.setWindowTitle("Tic Tac Toe Robot UI")
    stack.resize(920, 520)
    stack.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
