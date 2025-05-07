# SUAS 2022-2025 Project

This repository contains the code and resources for the UAVs at Berkeley team's SUAS (Student Unmanned Aerial Systems) competition project. The project focuses on developing autonomous flight control, computer vision, and mission planning systems for our competition drone.

## Project Structure

### Active Development Directories

- `SUAS_2025/` - Current active development directory containing:
  - Mission control and flight planning scripts
  - Computer vision and mapping algorithms
  - Vehicle state management
  - RTMP streaming capabilities
  - Key files:
    - `mission.py` - Main mission control logic
    - `mapping.py` - Aerial mapping implementation
    - `utils.py` - Common utility functions
    - `vehicle_state.py` - Drone state management

- `mapping_test/` - Testing and development for aerial mapping:
  - Image stitching and processing
  - Test image generation
  - Example implementations
  - Key files:
    - `image_stitcher.py` - Image stitching algorithm
    - `image_slicer.py` - Image processing utilities
    - `example_run.py` - Example usage

### External Dependencies (Git Submodules)

- `yolov5/` - Object detection system for target recognition
- `siyi_sdk/` - Camera control and interface SDK

### Electronics and Hardware

- `bearcopter_electronics/` - PCB design files and schematics:
  - Power train design
  - Communication systems
  - Computing hardware
  - Gimbal control
  - Payload integration

### Legacy and Reference

- `SUAS_2024/` - Previous year's implementation
- `fa24_106a/` - Course project work (EECS 106A) containing image processing algorithms

### Other Components

- `catkin_ws/` - ROS workspace (Note: Current status and usage unclear)
- `targets/` - Target detection and recognition resources. These are for previous year's stuff.
- `tests/` - Test suites and validation code

## Getting Started

1. Clone the repository with submodules:
```bash
git clone --recursive [repository-url]
```

2. Set up the development environment:
   - Python 3.x
   - Required Python packages (to be listed)
   - KiCad (for electronics development)

3. Key areas to focus on:
   - Mission control: `SUAS_2025/mission.py`
   - Mapping: `SUAS_2025/mapping.py`
   - Image processing: `mapping_test/`
   - Electronics: `bearcopter_electronics/`

## Development Guidelines

1. New features should be developed in the `SUAS_2025` directory
2. Mapping-related work should be tested in `mapping_test` first
3. All code should include appropriate documentation and tests

## Notes

- The `catkin_ws` directory appears to be a ROS workspace, but its current status and usage are unclear. Please verify with the team before using.
- Some directories may contain legacy code or work-in-progress features. Always check with the team about the current status of any component.

## Contributing

Please refer to the team's contribution guidelines and coding standards. All contributions should be properly documented and tested before being merged into the main codebase.