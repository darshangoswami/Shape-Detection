# Computer Vision Project - Edge Detection and Shape Recognition

## Overview

This project implements several fundamental computer vision algorithms including edge detection, corner detection, and shape recognition using the Hough transform. The implementation includes interactive boundary tracing and circle detection capabilities.

## Installation and Requirements

1. Download and extract the project.

2. In the project directory set up a virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

Main dependencies:

- Python 3.8+
- NumPy
- OpenCV (cv2)
- Matplotlib
- Pillow (PIL)
- SciPy

## Project Structure

```
.
├── problem1_1.py  # Gradient-based edge detection
├── problem1_2.py  # Interactive boundary tracing
├── problem2.py    # Harris corner detection
├── problem3_1.py  # Hough transform for line detection
├── problem3_2.py  # Hough transform for circle detection
└── problem_utils.py  # Utility functions
```

## Running the Code

### 1. Gradient-Based Edge Detection

```bash
python problem1_1.py
```

- Loads a grayscale image
- Applies Gaussian smoothing
- Computes gradient magnitude and orientation
- Displays results including edge visualization

### 2. Boundary Tracing

```bash
python problem1_2.py
```

- Interactive interface for selecting boundary points
- Click points on the image to trace object boundaries
- Press 'q' to finish and display results

### 3. Corner Detection

```bash
python problem2.py
```

- Implements Harris corner detector
- Processes test images
- Displays detected corners with visual markers

### 4. Hough Transform

For line detection:

```bash
python problem3_1.py
```

For circle detection:

```bash
python problem3_2.py
```

## Note

### After you run any of the files, a window will pop up with the resulting image.

### Once you're done interacting/viewing the results, press "q" to exit.
