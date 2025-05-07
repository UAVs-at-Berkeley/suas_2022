# Mapping Test Directory

This directory contains tools and utilities for testing and developing image stitching and mapping capabilities for UAV applications. 
The codebase is designed to handle image capture, processing, and stitching of overlapping aerial images.

## Directory Structure

- `image_stitcher.py`: Core image stitching implementation using SIFT features and affine transformations
- `image_slicer.py`: Utility to split large images into overlapping sections for testing
- `random_image_generator.py`: Tool to generate test images with random patterns
- `example_run.py`: Example script demonstrating the image capture and stitching workflow
- `image_capture_modified.py`: Modified image capture utility for UAV camera integration

### Test Image Directories
- `new test images/`: Contains recent test images
- `old test images/`: Archive of previous test images
- `random_overlap_images/`: Generated test images with random patterns
- `image_capture/`: Directory for python files related to live image capture on the drone

## Key Components

### Image Stitching (`image_stitcher.py`)
- Implements SIFT-based feature detection and matching
- Uses affine transformations for image alignment
- Handles overlapping regions between images
- Supports both horizontal and vertical image stitching
- Includes debug visualization capabilities

### Image Slicing (`image_slicer.py`)
- Splits large images into overlapping sections
- Configurable overlap ratios and grid dimensions
- Optional random variations in overlap regions
- Supports rotation for testing alignment robustness

### Random Image Generator (`random_image_generator.py`)
- Creates test images with random patterns
- Useful for testing stitching algorithms
- Configurable pattern size and density

### Example Run Script (`example_run.py`)
- Demonstrates the complete workflow
- Supports both live capture and test image processing
- Includes test cases for different grid configurations

## Usage

1. For live image capture and stitching:
   - Configure the RTSP URL in `example_run.py`
   - Run the test function in `example_run.py`

## Configuration

Key parameters can be modified in the respective files:
- Image dimensions and overlap ratios (`image_stitcher.py` - KING_SIZE, OVERLAP_X, OVERLAP_Y)
- SIFT feature detection parameters (`image_stitcher.py` - SIFT_create parameters)
- Grid dimensions for slicing (`image_slicer.py` - COLS, ROWS)
- Random variation settings (`image_slicer.py` - RANDOM_VARIATION, MAX_ROTATION)

## Dependencies
- OpenCV (cv2)
- NumPy
- imutils 