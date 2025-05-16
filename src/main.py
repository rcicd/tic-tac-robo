from arm_lib import DeviceConnection, BaseClient, BaseCyclicClient, move_to_home, set_gripper, cartesian_action_movement
from april_tag_lib import ArucoTracker, find_frame_with_home_marker, rvec_to_z_angle, capture_frame, markers_relative_to_home
from tic_tac_toe import find_best_move, check_winner
import time

BOARD = 3
CELL  = 0.05
MARKER_HALF = 0.0135

def pose_to_cell(pose):
    offset = 0.0135
    x = pose['x'] + offset
    y = pose['y'] + offset

    if x > 0 or y > 0:
        return None

    col = int((-x) // 0.05)
    row = int((-y) // 0.05)

    if 0 <= col < 3 and 0 <= row < 3:
        return row, col
    return None


def base_position(base, base_cyclic, dx=0, dy=0, dtz=0):
    cartesian_action_movement(base, base_cyclic,
                                x = -.05+dx, y =-.15+dy, z = -.4,
                                tx = 90, tz = dtz)
    # cartesian_action_movement(base, base_cyclic, tz=30)

def find_marker(
    tracker: ArucoTracker,
    *,
    marker_id: int = 0,
    home_id: int = 3,
    max_tries: int = 20,
    cam_index: int = 0,
):
    last_error = None

    for attempt in range(1, max_tries + 1):
        frame = capture_frame(cam_index)

        try:
            detections = tracker.detect_markers(frame)
            if not detections:
                raise ValueError("No markers detected in the frame.")

            try:
                home_pose = next(data for mid, data in detections if mid == home_id)
            except StopIteration:
                raise ValueError(f"Home marker id={home_id} not visible.")
            
            for mid, data in detections:
                if mid != marker_id:
                    continue

                rel_pose = ArucoTracker.relative_pose(home_pose, data)
                x = rel_pose["x"]
                y = rel_pose["y"]
                z = rel_pose["z"]
                yaw = rel_pose["yaw_deg"]    

                if x < 0 and y < 0:
                    continue

                print(
                    f"[INFO] Попытка {attempt}: marker {marker_id} is found "
                    f"x = {x:.2f} "
                    f"y = {y:.2f} "
                    f"z = {z:.2f} "
                    f"z_angle={yaw:.2f}°"
                )
                return x, y, z, yaw

            raise ValueError(f"Marker {marker_id} is not visible in the frame.")

        except ValueError as err:
            last_error = err
            print(f"[WARN] Try {attempt}/{max_tries}: {err}")
            time.sleep(0.05)

    raise RuntimeError(
        f"Marker {marker_id} not found in {max_tries} tries! "
        f"Last error: {last_error}"
    )

def read_board(frame, tracker, home_id=3):
    rel = markers_relative_to_home(frame, tracker, home_id=home_id)
    board = [['.' for _ in range(BOARD)] for _ in range(BOARD)]

    for mid, pose in rel:
        cell = pose_to_cell(pose)
        if cell is None:
            continue
        r, c = cell
        symbol = 'X' if mid == 0 else 'O'
        board[r][c] = symbol
    return board


if __name__ == "__main__":
    tracker = ArucoTracker()
    print("Welcome to Tic Tac Toe!")
    player = input("X/O?")
    assert player in ["X", "O"], "Invalid player. Choose 'X' or 'O'."

    robot_marker = 1 if player == "X" else 0
    print(f"Robot marker: {robot_marker}")

    calibration = input("Do you want to calibrate? (y/n)")
    with DeviceConnection.createTcpConnection() as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        tracker = ArucoTracker()

        move_to_home(base)

        if calibration == "y":
            move_to_home(base)
            base_position(base, base_cyclic, dy=0.08, dx=0.03)
            cartesian_action_movement(base, base_cyclic,
                                            tz = 25)
            set_gripper(base, 80)
            input("Place game field under the pointer and press Enter")
            move_to_home(base)
        
        if player == "X":
            input("make your move and press Enter")
        board = read_board(capture_frame(), tracker)
        print("Current board:")
        for row in board:
            print(" ".join(row))
        board = [[i.replace('.', '') for i in row] for row in board]
        while not (game_over := check_winner(board)):
            move_to_home(base)
            x, y, z, yaw = find_marker(tracker, 
                           cam_index=0, 
                           max_tries=20,
                           marker_id=robot_marker)
            board = read_board(capture_frame(), tracker)
            print("Current board:")
            for row in board:
                print(" ".join(row))
            board = [[i.replace('.', '') for i in row] for row in board]
            move = find_best_move(board, 'X' if player == 'O' else 'O')
            print(f"Best move: {move}")
            if move is not None:
                base_position(base, base_cyclic, dy=0.08, dx=0.03)
                cartesian_action_movement(base, base_cyclic,
                                                tz = 25)
                cartesian_action_movement(base, base_cyclic,
                                          x = x, y = y, tz = yaw)
                set_gripper(base, 40)
                cartesian_action_movement(base, base_cyclic,
                                          z = -.05)
                set_gripper(base, 80)
                cartesian_action_movement(base, base_cyclic,
                                          z = .1)
                move_to_home(base)
                base_position(base, base_cyclic, dy=0.08, dx=0.03)
                cartesian_action_movement(base, base_cyclic,
                                          tz = 25)
                cartesian_action_movement(base, base_cyclic,
                                            x = -(0.0135 + 0.05 * move[1]+0.025))
                cartesian_action_movement(base, base_cyclic,
                                            y = -(0.0135 + 0.05 * move[0]+0.025))
                
                row, col = move
                board[row][col] = 'X' if player == 'O' else 'O'
                print(f"Move made at: {row}, {col}")
                cartesian_action_movement(base, base_cyclic,
                                          z = -.05)
                set_gripper(base, 40)
                move_to_home(base)
                if game_over := check_winner(board):
                    print(f"Game over: {game_over}")
                    exit()
                input("Press Enter after placing the marker")
        else:
            print(f"Game over: {game_over}")
            exit()
