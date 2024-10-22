# problem3_2.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    """
    Load and convert image to grayscale numpy array.
    """
    return np.array(Image.open(path).convert('L'))

def myHoughCircleTrain(imBW, c, ptlist):
    """
    Train the Hough transform for circle detection.
    
    Args:
    imBW: Binary input image containing a single circular object
    c: Reference point (center of the circle)
    ptlist: Ordered list of boundary points
    
    Returns:
    yourcellvar: Cell array containing necessary data for circle detection
    """
    # Calculate radius
    radii = [np.sqrt((x - c[0])**2 + (y - c[1])**2) for x, y in ptlist]
    radius = np.mean(radii)
    
    # Create a template for circle detection
    theta = np.linspace(0, 2*np.pi, 100)
    x_template = radius * np.cos(theta)
    y_template = radius * np.sin(theta)
    
    # Store the template and radius in a cell array
    yourcellvar = {
        'radius': radius,
        'x_template': x_template,
        'y_template': y_template
    }
    
    return yourcellvar

def myHoughCircleTest(imBWnew, yourcellvar):
    """
    Test the Hough transform for circle detection on a new image.
    
    Args:
    imBWnew: New binary input image to detect circles in
    yourcellvar: Cell array containing data from training
    
    Returns:
    centers: List of detected circle centers (top 2)
    """
    radius = yourcellvar['radius']
    x_template = yourcellvar['x_template']
    y_template = yourcellvar['y_template']
    
    height, width = imBWnew.shape
    accumulator = np.zeros((height, width))
    
    # Perform Hough transform
    edge_points = np.argwhere(imBWnew > 0)
    for y, x in edge_points:
        for dx, dy in zip(x_template, y_template):
            a = int(x - dx)
            b = int(y - dy)
            if 0 <= a < width and 0 <= b < height:
                accumulator[b, a] += 1
    
    # Find the top 2 peaks in the accumulator
    centers = []
    for _ in range(2):
        idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        centers.append((idx[1], idx[0]))  # (x, y) format
        # Suppress this maximum and its neighborhood
        y, x = idx
        y_min, y_max = max(0, y - 10), min(height, y + 11)
        x_min, x_max = max(0, x - 10), min(width, x + 11)
        accumulator[y_min:y_max, x_min:x_max] = 0
    
    return centers

def plot_results(image, centers, radius):
    """
    Plot the original image with detected circles.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(image, cmap='gray')
    plt.title('Detected Circles')
    plt.axis('off')
    
    for center in centers:
        circle = plt.Circle(center, radius, fill=False, color='r')
        plt.gca().add_artist(circle)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Running Hough circle detection...")
    
    try:
        # Training phase
        print("Training phase...")
        train_image_path = './train.png'
        train_image = load_image(train_image_path)
        train_binary = (train_image > 128).astype(np.uint8)
        
        # Example center and boundary points (replace with actual values)
        center = (100, 100)
        boundary_points = [(110, 100), (100, 110), (90, 100), (100, 90)]
        
        trained_data = myHoughCircleTrain(train_binary, center, boundary_points)
        print("Training complete!")
        
        # Testing phase
        print("\nTesting phase...")
        test_image_path = './test.png'
        test_image = load_image(test_image_path)
        test_binary = (test_image > 128).astype(np.uint8)
        
        detected_centers = myHoughCircleTest(test_binary, trained_data)
        
        # Plot results
        plot_results(test_image, detected_centers, trained_data['radius'])
        
        # Print results
        print("\nDetected circle centers:")
        for i, center in enumerate(detected_centers, 1):
            print(f"Circle {i}: center at {center}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find image file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()