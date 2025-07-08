from arm_lib import DeviceConnection, BaseClient, BaseCyclicClient, move_to_home, set_gripper, cartesian_action_movement
from aprli_tag_new import ArucoTracker, markers_relative_to_home, capture_image
from tic_tac_toe import find_best_move, check_winner
import time

BOARD = 3
CELL = 0.08
FIELD_SIZE = CELL * BOARD
MARKER_SIZE = 0.04
TILE_SIZE = 0.05
HOME_SHIFT = 0.024

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
        return 'X'
    elif marker_id > 0 and marker_id % 2 == 0:
        return 'O'
    return None

def find_markers_by_type(detections, marker_type):
    markers = []
    from pprint import pprint
    pprint(detections)
    input()
    for mid, data in detections:
        print(mid)
        if get_marker_type(mid) == marker_type:
            markers.append((mid, data))
    return markers


def base_position(base, base_cyclic, dx=0, dy=0, dtz=0):
    cartesian_action_movement(base, base_cyclic,
                                x = -.05+dx, y =-.15+dy, z = -.4,
                                tx = 90, tz = dtz)

def find_marker_by_type(
    tracker: ArucoTracker,
    *,
    marker_type: str,  # 'X' or 'O'
    home_id: int = 0,
    max_tries: int = 20,
):
    last_error = None

    for attempt in range(1, max_tries + 1):
        frame = capture_image()

        try:
            detections = tracker.detect_markers(frame)
            if not detections:
                raise ValueError("No markers detected in the frame.")

            try:
                home_pose = next(data for mid, data in detections if mid == home_id)
            except StopIteration:
                raise ValueError(f"Home marker id={home_id} not visible.")
            for mid, data in detections:
                # print(mid)
                # print(get_marker_type(mid))
                # print(marker_type)
                # print(ord(marker_type))
                # print(ord(get_marker_type(mid)))
                # print(marker_type == get_marker_type(mid))
                # input()
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
                    f"[INFO] Attempt {attempt}: marker {mid} ({marker_type}) is found "
                    f"x = {x:.3f} "
                    f"y = {y:.3f} "
                    f"z = {z:.3f} "
                    f"z_angle={yaw:.1f}°"
                )
                return mid, x, y, z, yaw

            raise ValueError(f"Marker type {marker_type} is not visible in the frame.")

        except ValueError as err:
            last_error = err
            print(f"[WARN] Attempt {attempt}/{max_tries}: {err}")
            time.sleep(0.05)

    raise RuntimeError(
        f"Marker type {marker_type} not found in {max_tries} tries! "
        f"Last error: {last_error}"
    )

def read_board(frame, tracker, home_id=0):
    from aprli_tag_new import get_session_logger
    logger = get_session_logger()
    logger.logger.info(f"Reading board state with home marker id={home_id}")
    
    logger.save_image(frame, "board_reading_input.jpg")
    
    rel = markers_relative_to_home(frame, tracker, home_id=home_id)
    board = [['.' for _ in range(BOARD)] for _ in range(BOARD)]
    
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
            detected_markers.append({
                "marker_id": mid,
                "marker_type": marker_type,
                "position": {"row": r, "col": c},
                "pose": pose
            })
            logger.logger.debug(f"Placed marker {mid} ({marker_type}) at position ({r}, {c})")
        else:
            logger.logger.debug(f"Marker {mid} has unknown type")
    
    board_state = {
        "timestamp": logger.logger.handlers[0].formatter.formatTime(logger.logger.makeRecord(
            logger.logger.name, logger.logger.level, "", 0, "", (), None)),
        "home_marker_id": home_id,
        "board_size": BOARD,
        "cell_size": CELL,
        "detected_markers": detected_markers,
        "board_state": board
    }
    logger.save_data(board_state, f"board_state_{int(time.time())}")
    
    logger.logger.info(f"Board reading completed. Detected {len(detected_markers)} markers on board:")
    for row_idx, row in enumerate(board):
        logger.logger.info(f"Row {row_idx}: {' '.join(row)}")
    
    return board

def loop_read_board():
    tracker = ArucoTracker(marker_size=MARKER_SIZE, ema_alpha=1.0)  
    while True:
        input("Press Enter to capture the board state...")
        board = read_board(capture_image(), tracker)
        for row in board:
            print(" ".join(row))

if __name__ == "__main__":
    # loop_read_board()
    # exit(0)
    print("Welcome to Tic Tac Robo!")
    player = input("Choose your marker (X or O): ").strip().upper()
    assert player in ["X", "O"], "Wrong marker! Choose X or O."

    robot_symbol = 'O' if player == 'X' else 'X'
    print(f"Robot's symbol: {robot_symbol}")

    calibration = input("Do you want to calibrate (y/n): ")
    
    with DeviceConnection.createTcpConnection() as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        tracker = ArucoTracker(marker_size=MARKER_SIZE
                            , ema_alpha=1)

        move_to_home(base)

        if calibration == "y":
            move_to_home(base)
            base_position(base, base_cyclic, dy=0.08, dx=0.03)
            cartesian_action_movement(base, base_cyclic, tz=25)
            set_gripper(base, 80)
            input("Put the home marker below the arm and press Enter")
            move_to_home(base)
        
        if player == "X":
            input("Place your marker on the board and press Enter to start")

        board = read_board(capture_image(), tracker)
        print("Current board state:")
        for row in board:
            print(" ".join(row))
        board = [[cell.replace('.', '') for cell in row] for row in board]
    
        # time.sleep(5)
    
        while not (game_over := check_winner(board)):
            move_to_home(base)
            
            try:
                marker_id, x, y, z, yaw = find_marker_by_type(
                    tracker, 
                    marker_type=robot_symbol,
                    max_tries=20
                )
                print(f"Marker is found {marker_id} of type {robot_symbol}")
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                break
            
            img = capture_image()
            
            from aprli_tag_new import get_session_logger
            logger = get_session_logger()
            logger.save_image(img, "board_state_analysis.jpg")
            logger.logger.info("Captured image for board state analysis")
            
            board_ = read_board(img, tracker)
            print("Current board state:")
            for row in board_:
                print(" ".join(row))
            
            board_ = [[cell.replace('.', '') for cell in row] for row in board_]
            
            move = find_best_move(board_, robot_symbol)
            print(f"Best move: {move}")
            
            board = board_
            
            if move is not None:
                base_position(base, base_cyclic, dy=0.08, dx=0.03)
                cartesian_action_movement(base, base_cyclic, tz=25)
                cartesian_action_movement(base, base_cyclic, x=x, y=y, tz=180-abs(yaw))

                set_gripper(base, 40)
                cartesian_action_movement(base, base_cyclic, z=-.05)
                set_gripper(base, 70)
                cartesian_action_movement(base, base_cyclic, z=.1)
                
                move_to_home(base)
                base_position(base, base_cyclic, dy=0.08, dx=0.03)
                cartesian_action_movement(base, base_cyclic, tz=25)
                
                row, col = move
                target_x = -(HOME_SHIFT + CELL * col + CELL / 2)
                target_y = -(HOME_SHIFT + CELL * row + CELL / 2)

                cartesian_action_movement(base, base_cyclic, x=target_x)
                cartesian_action_movement(base, base_cyclic, y=target_y)
                
                cartesian_action_movement(base, base_cyclic, z=-.03)
                set_gripper(base, 50)
                move_to_home(base)

                board[row][col] = robot_symbol
                print(f"Turn made there: {row}, {col}")

                if game_over := check_winner(board):
                    print(f"Game over: {game_over}")
                    break
                
                input("Your turn! Place your marker and press Enter to continue.")
            else:
                print("No valid move found for the robot.")
                break
        
        print(f"Game over: {game_over}")
