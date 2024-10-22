import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

class ImageProcessor:
    """Handles image loading and edge detection"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return image

    @staticmethod
    def detect_edges(image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection with preprocessing"""
        edges = cv2.Canny(image, 100, 200)  # Adjusted thresholds
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

class BoundaryTracer:
    """Handles the tracing of object boundaries"""
    
    def __init__(self, edges: np.ndarray, original_image: np.ndarray):
        self.edges = edges
        self.display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        self.boundaries = []
        
    def trace_boundary(self, start_x: int, start_y: int) -> List[Tuple[int, int]]:
        """Trace boundary using Moore-Neighbor algorithm"""
        height, width = self.edges.shape
        traced_boundary = []
        visited = np.zeros_like(self.edges, dtype=bool)

        directions = [
            (-1, 0),  # N
            (-1, 1),  # NE
            (0, 1),   # E
            (1, 1),   # SE
            (1, 0),   # S
            (1, -1),  # SW
            (0, -1),  # W
            (-1, -1)  # NW
        ]

        current_point = (start_x, start_y)
        traced_boundary.append(current_point)
        visited[start_y, start_x] = True
        backtrack_direction = 0

        while True:
            found_next = False

            for i in range(8):
                idx = (backtrack_direction + i) % 8
                dx, dy = directions[idx]
                x, y = current_point[0] + dx, current_point[1] + dy

                if 0 <= x < width and 0 <= y < height:
                    if self.edges[y, x] != 0 and not visited[y, x]:
                        current_point = (x, y)
                        traced_boundary.append(current_point)
                        visited[y, x] = True
                        backtrack_direction = (idx + 5) % 8
                        found_next = True
                        break

            if not found_next or current_point == (start_x, start_y) or len(traced_boundary) > 10000:
                break

        return traced_boundary if len(traced_boundary) > 1 else []

    def update_display(self, boundary: List[Tuple[int, int]]):
        """Update display with new boundary"""
        if boundary:
            # Mark start point
            cv2.circle(self.display_image, boundary[0], 3, (0, 255, 0), -1)
            
            # Draw boundary
            for i in range(len(boundary) - 1):
                pt1 = boundary[i]
                pt2 = boundary[i + 1]
                cv2.line(self.display_image, pt1, pt2, (0, 0, 255), 2)
            
            self.boundaries.append(boundary)
            cv2.imshow('Boundary Tracer', self.display_image)

    def show_final_result(self, original_image: np.ndarray):
        """Show final result using matplotlib"""
        plt.figure(figsize=(12, 6))
        
        # Original image with boundaries
        plt.subplot(121)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image with Traced Boundaries')
        for boundary in self.boundaries:
            if boundary:
                x_coords, y_coords = zip(*boundary)
                plt.plot(x_coords, y_coords, 'r-', linewidth=2)
        plt.axis('off')
        
        # Edge map with boundaries
        plt.subplot(122)
        plt.imshow(self.edges, cmap='gray')
        plt.title('Edge Map with Traced Boundaries')
        for boundary in self.boundaries:
            if boundary:
                x_coords, y_coords = zip(*boundary)
                plt.plot(x_coords, y_coords, 'r-', linewidth=2)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

class InteractiveBoundaryTracer:
    """Main class for interactive boundary tracing"""
    
    def __init__(self, image_path: str):
        self.original_image = ImageProcessor.load_image(image_path)
        _, self.binary_image = cv2.threshold(self.original_image, 127, 255, cv2.THRESH_BINARY)
        self.edges = ImageProcessor.detect_edges(self.binary_image)
        self.tracer = BoundaryTracer(self.edges, self.original_image)

    def run(self):
        """Run the interactive boundary tracing session"""
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Seed point selected at: x={x}, y={y}")
                boundary = self.tracer.trace_boundary(x, y)
                if boundary:
                    self.tracer.update_display(boundary)

        cv2.namedWindow('Boundary Tracer')
        cv2.setMouseCallback('Boundary Tracer', on_mouse_click)
        cv2.imshow('Boundary Tracer', self.tracer.display_image)
        
        print("Click points to trace boundaries. Press 'q' to finish.")
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        self.tracer.show_final_result(self.original_image)

def main():
    try:
        tracer = InteractiveBoundaryTracer('./1.png')
        tracer.run()
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()