# Tic-Tac-Robo: Robot-Controlled Tic-Tac-Toe with Computer Vision

[Demo](https://youtu.be/N_QuOAMHb9M)

This project combines robotics, computer vision, and game AI to let a Kinova Gen3 Lite robotic arm play Tic-Tac-Toe using AprilTag (or ArUco) markers. The system recognizes the game field, reads marker positions, calculates optimal moves, and controls the robot arm to make its own moves.

## Features

- **Automatic camera calibration** using chessboard images.
- **AprilTag or ArUco marker detection** for field recognition.
- **Game state detection** from real marker positions.
- **Minimax algorithm** for optimal Tic-Tac-Toe AI.
- **Robot arm control** (Kinova Gen3 Lite) for marker placement.
- **Gripper control** for manipulating physical markers.
- **Comprehensive logging system** with automatic session management.
- **Image saving** for all captured and processed frames.
- **JSON data export** for detection results and analysis.
- Clear error handling and interactive prompts.

## Session Logging

The system now includes comprehensive logging capabilities:

- **Automatic session directories**: Each run creates a new numbered session directory in `logs/`
- **Multi-level logging**: DEBUG, INFO, WARNING, ERROR messages to both file and console
- **Image capture**: All original, processed, and annotated images are saved
- **Data export**: Detection results, poses, and calibration data saved as JSON
- **Performance profiling**: Execution time tracking for all major functions

See `logs/README.md` for detailed information about the logging structure.

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
- `ui.py` — User interface for game interaction.
- `calibration_images/` — Calibration images for the camera.
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

## Setup Instructions

### ALL MEANT ABOVE SHOULD BE DONE ON DEVICE THAT HAVE CAMERA AND ROBOT ARM CONNECTED

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rcicd/tic-tac-robo.git
   cd tic-tac-robo
   ```
2. **Install dependencies:**
   * You should install [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to manage Python versions and virtual environments
   * Download required [python package](https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl) from kortex artifactory and put it in root of project
3. Run following commands:
   ```bash
   make setup
   make install
   ```
4. print on A3 this [game board](masterpiece.png)
## Usage

### 1. Camera Calibration

Before starting, calibrate your camera for best results:
```
make calibrate
```
### 2. Start the Game
Run the main program and follow on-screen instructions:
```bash
make run
```
- Ensure the robot arm is powered on and connected
- Open the web interface in your browser (default: http://localhost:8000 or http://\<duckie-name\>.local:8000 for Raspberry Pi)
- Press "Configure", arm will move to some position, 
place the printed game board positioning it's 
cross under arm pointer. When you are ready, press
"OK"
- Press "Play" to start a new game
- Place your move and "Turn is done"
- The robot will detect the game state and play its move
- Repeat until the game ends

### 3. Playing
- The robot reads the board state automatically between moves
- Markers are placed using the robotic arm
- The board size and tag sizes are assumed to be known and constant (default: cells 8x8 cm, tag size 40 mm)
