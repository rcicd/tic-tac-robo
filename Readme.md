# Tic-Tac-Robo: Robot-Controlled Tic-Tac-Toe with Computer Vision

This project combines robotics, computer vision, and game AI to let a Kinova Gen3 Lite robotic arm play Tic-Tac-Toe using AprilTag (or ArUco) markers. The system recognizes the game field, reads marker positions, calculates optimal moves, and controls the robot arm to make its own moves.

## Features

- **Automatic camera calibration** using chessboard images.
- **AprilTag or ArUco marker detection** for field recognition.
- **Game state detection** from real marker positions.
- **Minimax algorithm** for optimal Tic-Tac-Toe AI.
- **Robot arm control** (Kinova Gen3 Lite) for marker placement.
- **Gripper control** for manipulating physical markers.
- Clear error handling and interactive prompts.

## Hardware and Software Requirements

- Kinova Gen3 Lite robot arm
- Raspberry Pi (or any Linux computer compatible with your camera)
- Camera compatible with `rpicam-jpeg` or OpenCV
- Printed AprilTag or ArUco markers (one per cell + one reference/home marker)
- Python 3.8+
- Packages: `opencv-python`, `numpy`, `kortex_api` (Kinova API Python bindings)

## Project Structure

- `arm_lib.py` — Robot arm and gripper control functions (Kinova API).
- `april_tag_lib.py` — Marker detection and camera calibration routines.
- `tic_tac_toe.py` — Game logic: minimax algorithm, win check, move selection.
- `main.py` — Demo script: robot, camera, and game integration.
- `calib_imgs/` — Calibration images for the camera.
- `README.md` — This documentation.

## How It Works

1. **Camera Calibration:**  
   The script captures a series of chessboard images for camera calibration using OpenCV. This is recommended for accurate marker detection.

2. **Game Loop:**  
   - The robot moves to a safe "home" position.
   - The camera captures the board, detects marker positions, and determines the current game state.
   - The human player moves first or second (configurable at start).
   - The robot calculates its optimal move using the minimax algorithm and performs the necessary motions to place its marker using the gripper.
   - The process repeats until the game is finished (win/draw).

## Usage

### 1. Camera Calibration

Before starting, calibrate your camera for best results:
```bash
python do_samples.py
python calibrate_camera.py
```
### 2. Start the Game
Run the main program and follow on-screen instructions:
```bash
python main.py
```
- Select your marker (X or O)
- Place your move and press Enter when prompted
- The robot will detect the game state and play its move

### 3. Playing
- The robot reads the board state automatically between moves

- Markers are placed using the robotic arm

- The board size and tag sizes are assumed to be known and constant (default: cells 5x5 cm, tag size 27 mm)