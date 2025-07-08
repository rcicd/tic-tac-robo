# ArUco Detection Logs

This directory contains session logs for ArUco marker detection runs.

## Session Directory Structure

Each session creates a numbered directory (e.g., `session_001`, `session_002`, etc.) with the following structure:

```
session_XXX/
├── session.log                 # Main log file with all debug/info messages
├── images/                     # All captured and processed images
│   ├── XXXX_original_capture.jpg     # Original camera capture
│   ├── XXXX_preprocessed_capture.jpg # Preprocessed capture
│   ├── XXXX_input_frame.jpg          # Input frame for detection
│   └── XXXX_annotated_frame.jpg      # Final annotated frame with markers
├── debug_images/               # Debug/intermediate images
│   ├── XXXX_preprocessed_frame.jpg   # Preprocessed frame
│   └── XXXX_detection_input.jpg      # Detection input frame
└── data/                       # JSON data files
    ├── calibration_data.json         # Camera calibration parameters
    ├── detection_TIMESTAMP.json      # Raw detection results
    ├── relative_poses_TIMESTAMP.json # Relative poses data
    └── session_summary.json          # Session summary with final results
```

## Log Levels

- **INFO**: General session flow and important events
- **DEBUG**: Detailed processing information, measurements, and intermediate results
- **WARNING**: Non-critical issues (e.g., failed marker detection)
- **ERROR**: Critical errors that may affect the session

## Data Files

### calibration_data.json
Contains camera calibration parameters and detector configuration.

### detection_TIMESTAMP.json
Raw detection results for each marker, including:
- Rotation vectors (rvec)
- Translation vectors (tvec, both raw and filtered)
- Orientation in radians and degrees
- Distance measurements

### relative_poses_TIMESTAMP.json
Relative pose data between detected markers and the home marker.

### session_summary.json
High-level summary of the session including:
- Total markers detected
- Home marker ID
- Final relative positions and orientations

## Image Naming Convention

Images are numbered incrementally with 4-digit prefixes (e.g., `0001_`, `0002_`) to maintain processing order.

## Timestamps

All timestamps are in ISO format: `YYYY-MM-DDTHH:MM:SS.ffffff`
