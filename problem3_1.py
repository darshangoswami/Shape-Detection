import numpy as np
import matplotlib.pyplot as plt
from problem_utils import load_image
from problem1_1 import gradient_edge_detector
import cv2

def edge_detection(image, low_threshold=0.1, high_threshold=0.2):
    magnitude, _ = gradient_edge_detector(image, sigma=1.0)
    low_mask = magnitude > low_threshold
    high_mask = magnitude > high_threshold
    edge_map = ((low_mask & high_mask) | high_mask).astype(np.uint8)
    return edge_map

def myHoughLine(imBW, n, rho_res=1, theta_res=0.5):
    """
    Implement Hough transform to detect line segments.
    """
    height, width = imBW.shape
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    y_idxs, x_idxs = np.nonzero(imBW)
    
    for i, j in zip(x_idxs, y_idxs):
        for theta_idx in range(len(thetas)):
            rho = int((i * cos_t[theta_idx] + j * sin_t[theta_idx]) / rho_res) + diag_len
            if 0 <= rho < len(rhos):
                accumulator[rho, theta_idx] += 1
    
    lines = []
    for _ in range(n):
        idx = np.argmax(accumulator)
        rho_idx, theta_idx = np.unravel_index(idx, accumulator.shape)
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
        accumulator[max(0, rho_idx-5):min(accumulator.shape[0], rho_idx+6),
                   max(0, theta_idx-5):min(accumulator.shape[1], theta_idx+6)] = 0
    
    return lines

def find_line_segments(image, lines, min_length=20):
    segments = []
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        mask = np.zeros_like(image)
        cv2.line(mask, (x1, y1), (x2, y2), 1, 1)
        
        intersection = np.logical_and(mask, image)
        points = np.argwhere(intersection)
        
        if len(points) >= 2:
            dist = np.linalg.norm(points[-1] - points[0])
            if dist >= min_length:
                segments.append((tuple(points[0][::-1]), tuple(points[-1][::-1])))
    
    return segments

def plot_line_segments(image, segments):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(image, cmap='gray')
    plt.title('Detected Line Segments')
    plt.axis('off')
    
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'r-')
    
    plt.tight_layout()
    plt.show()


def main():
    print("Running Hough line detection...")
    
    # Load image
    image_path = './3.png'
    try:
        image = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image file at {image_path}")
        return
    
    # Apply edge detection
    print("Detecting edges...")
    edge_image = edge_detection(image, low_threshold=0.01, high_threshold=0.03)
    
    # Detect lines using Hough transform
    print("Applying Hough transform...")
    n_lines = 5
    detected_lines = myHoughLine(edge_image, n_lines)
    
    # Find line segments
    print("Finding line segments...")
    line_segments = find_line_segments(edge_image, detected_lines)
    
    # Plot results
    plot_line_segments(image, line_segments)
    
    # Print results
    print(f"\nDetected {len(line_segments)} line segments:")
    for i, ((x1, y1), (x2, y2)) in enumerate(line_segments, 1):
        print(f"Segment {i}: ({x1}, {y1}) to ({x2}, {y2})")

if __name__ == "__main__":
    main()