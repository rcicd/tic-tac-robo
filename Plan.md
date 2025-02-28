# Project Plan: Robotic Arm for Tic-Tac-Toe 

## 1. Preparation Phase

### 1.1. Hardware
- **Robotic Arm:** Kinova Gen3 lite
- **Camera:** rpi camera module
- **Game Board:** A board marked with printed AprilTag markers corresponding to each cell

### 1.2. Software and Libraries
- **Programming Language:** Python 3.12
- **Libraries:**
  - OpenCV (for video capture and image processing)
  - OpenCV’s ArUco module configured for AprilTag
  - KINOVA KORTEX API for Python (for robotic arm control)
  - Additional libraries: numpy, argparse, etc.

### 1.3. Environment Setup
- Install Python and create a virtual environment
- Install required packages
- Configure connection to the robotic arm

## 2. Image Capture and Processing

### 2.1. Video Capture
- Use OpenCV to capture video stream from the camera

### 2.2. AprilTag Detection
- Convert captured images to grayscale
- Initialize the AprilTag detector with appropriate settings
- Detect markers and extract coordinates of their corners, center, and identifier
- Visualize detection results by drawing boundaries, centers, and marker IDs on the image

## 3. Tic-Tac-Toe Game Logic

### 3.1. Game Board Initialization
- Represent the game board as a 3×3 grid
- Map each detected marker (or its position) to a specific cell on the board

### 3.2. Game Flow Management
- Define turn order (player vs. robot)
- Analyze detected markers to determine the selected cell
- Robot selects its move using a strategy

### 3.3. Board Update
- Update the internal representation of the board after each move
- Display the current game state

## 4. Kinova Robotic Arm Control

### 4.1. Connection Initialization
- Use [api](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L)
- Establish connection to the robotic arm using configuration from 1.3

### 4.2. Motion Programming
- Define coordinates for moving the arm to the corresponding board cells
- Create functions to send movement commands via the high-level API

### 4.3. Integration with Game Logic
- Link game logic (cell selection) with robotic arm movement commands
- Implement the robot’s move: moving the arm to place its marker on the board

## 5. Component Integration

### 5.1. Main Application Loop
- Combine video processing, AprilTag detection, game logic, and robotic arm control into one main loop
- Implement error handling and proper shutdown procedures

### 5.2. Debugging and Testing
- Test AprilTag detection under various lighting conditions
- Test robotic arm movements in a safe mode
- Perform integration testing of the complete system

## 6. Conclusion
- Evaluate the performance of the system
- Document any issues
- Finally play some tic-tac-toe with the robotic arm and submit the project
