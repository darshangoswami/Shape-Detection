import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def myHoughCircleTrain(imBW, c, ptlist):
    # Calculate relative positions of boundary points from center
    rel_positions = [(x - c[0], y - c[1]) for x, y in ptlist]
    
    # Calculate radius for verification
    radii = [np.sqrt(dx*dx + dy*dy) for dx, dy in rel_positions]
    radius = np.mean(radii)
    
    # Store only necessary information
    yourcellvar = {
        'rel_positions': rel_positions,
        'radius': radius
    }
    
    return yourcellvar

def myHoughCircleTest(imBWnew, yourcellvar):
    height, width = imBWnew.shape
    rel_positions = yourcellvar['rel_positions']
    
    # Create accumulator array
    accumulator = np.zeros((height, width))
    
    # Get edge points
    edge_points = np.argwhere(imBWnew > 0)
    
    # Vote in accumulator space
    for y, x in edge_points:
        for dx, dy in rel_positions:
            center_x = x - dx
            center_y = y - dy
            if (0 <= center_x < width and 0 <= center_y < height):
                accumulator[center_y, center_x] += 1
    
    # Find top 2 centers using non-maximum suppression
    centers = []
    min_distance = int(yourcellvar['radius'] * 0.8)
    
    for _ in range(2):  # Find exactly 2 centers as required
        if np.max(accumulator) == 0:  # No more peaks
            break
            
        y, x = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        centers.append((x, y))  # Store in (x,y) format
        
        # Suppress region around detected center
        y_min = max(0, y - min_distance)
        y_max = min(height, y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(width, x + min_distance + 1)
        accumulator[y_min:y_max, x_min:x_max] = 0
    
    return centers

def visualize_results(image, centers, radius):
    """Helper function to visualize results."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Detected Circles')
    
    for center in centers:
        circle = plt.Circle(center, radius, fill=False, color='r', linewidth=2)
        plt.gca().add_artist(circle)
        plt.plot(center[0], center[1], 'r+', markersize=10)
    
    plt.axis('off')
    plt.show()

# Example usage
def main():
    # Use the provided center and boundary points
    c = (100, 100)
    ptlist = [
          (90, 97), (90, 98), (90, 99), (90, 100), (90, 101), (90, 102), (90, 103),
          (91, 95), (91, 96), (91, 104), (91, 105), (92, 94), (92, 106), (93, 93),
          (93, 107), (94, 92), (94, 108), (95, 91), (95, 109), (96, 91), (96, 109),
          (97, 90), (97, 110), (98, 90), (98, 110), (99, 90), (99, 110), (100, 90),
          (100, 110), (101, 90), (101, 110), (102, 90), (102, 110), (103, 90),
          (103, 110), (104, 91), (104, 109), (105, 91), (105, 109), (106, 92),
          (106, 108), (107, 93), (107, 107), (108, 94), (108, 106), (109, 95),
          (109, 96), (109, 104), (109, 105), (110, 97), (110, 98), (110, 99),
          (110, 100), (110, 101), (110, 102), (110, 103)
    ]
    
    # Load images
    train_img = np.array(Image.open('./train.png').convert('L'))
    test_img = np.array(Image.open('./test.png').convert('L'))
    
    # Convert to binary
    train_binary = (train_img > 128).astype(np.uint8)
    test_binary = (test_img > 128).astype(np.uint8)
    
    # Training phase
    trained_data = myHoughCircleTrain(train_binary, c, ptlist)
    
    # Testing phase
    centers = myHoughCircleTest(test_binary, trained_data)
    
    # Print results
    print("Detected circle centers:")
    for i, center in enumerate(centers, 1):
        print(f"Circle {i}: center at {center}")
    
    # Visualize results
    visualize_results(test_binary, centers, trained_data['radius'])

if __name__ == "__main__":
    main()