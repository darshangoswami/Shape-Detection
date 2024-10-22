import numpy as np
import matplotlib.pyplot as plt
from problem_utils import create_gaussian_kernel, apply_convolution, load_image

def gradient_edge_detector(image, sigma):
    """
    Perform gradient-based edge detection.
    """
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Smooth the image
    smoothed = apply_convolution(image, gaussian_kernel)
    
    # Compute derivatives
    dx_kernel = np.array([[-1, 0, 1]])
    dy_kernel = np.array([[-1], [0], [1]])
    
    grad_x = apply_convolution(smoothed, dx_kernel)
    grad_y = apply_convolution(smoothed, dy_kernel)
    
    # Ensure same shape
    min_shape = (min(grad_x.shape[0], grad_y.shape[0]), 
                min(grad_x.shape[1], grad_y.shape[1]))
    grad_x = grad_x[:min_shape[0], :min_shape[1]]
    grad_y = grad_y[:min_shape[0], :min_shape[1]]
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    
    return magnitude, orientation

def plot_results(image, magnitude, orientation):
    """
    Plot the results of edge detection.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    step = 10
    y, x = np.mgrid[step//2:magnitude.shape[0]:step, 
                    step//2:magnitude.shape[1]:step]
    u = np.cos(orientation[y, x])
    v = np.sin(orientation[y, x])
    plt.quiver(x, y, u, v, color='r', angles='xy', 
              scale_units='xy', scale=0.1)
    plt.title('Gradient Orientation')
    plt.gca().invert_yaxis()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Example usage
    print("Running gradient-based edge detection...")
    
    # Load image
    image_path = './1.png'
    try:
        image = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image file at {image_path}")
        return
        
    # Set parameter
    sigma = 1.0
    print(f"Processing with sigma = {sigma}")
    
    # Apply gradient-based edge detection
    magnitude, orientation = gradient_edge_detector(image, sigma)
    
    # Plot results
    plot_results(image, magnitude, orientation)

if __name__ == "__main__":
    main()