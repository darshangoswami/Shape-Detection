from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path
import numpy.typing as npt

@dataclass
class ImageData:
    """Container for image data and related information"""
    array: npt.NDArray[np.float32]
    path: Path
    
    @classmethod
    def from_path(cls, path: str) -> ImageData:
        """Create ImageData from an image file path"""
        img_path = Path(path)
        img_array = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        return cls(array=img_array, path=img_path)

class ConvolutionOperator:
    """Handles image convolution operations"""
    
    @staticmethod
    def apply(image: npt.NDArray[np.float32], 
             kernel: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply convolution with given kernel"""
        img_height, img_width = image.shape
        k_height, k_width = kernel.shape
        
        # Padding calculations
        pad_h = k_height // 2
        pad_w = k_width // 2
        
        # Pad the image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Output initialization
        output = np.zeros_like(image, dtype=np.float32)
        
        # Flip kernel for convolution
        kernel_flipped = np.flipud(np.fliplr(kernel))
        
        # Perform convolution
        for y in range(img_height):
            for x in range(img_width):
                region = padded[y:y+k_height, x:x+k_width]
                output[y, x] = np.sum(region * kernel_flipped)
                
        return output

class KernelFactory:
    """Factory class for creating various kernels"""
    
    @staticmethod
    def gaussian(size: int, sigma: float = 1.0) -> npt.NDArray[np.float32]:
        """Create a Gaussian kernel"""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2)/(2 * sigma**2))
                
        return kernel / np.sum(kernel)
    
    @staticmethod
    def sobel() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Create Sobel kernels for x and y directions"""
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
                           
        sobel_y = np.array([[ 1,  2,  1],
                           [ 0,  0,  0],
                           [-1, -2, -1]], dtype=np.float32)
                           
        return sobel_x, sobel_y

class HarrisDetector:
    """Implementation of Harris corner detection algorithm"""
    
    def __init__(self, sigma: float = 1.0, k: float = 0.04, window_size: int = 3):
        self.sigma = sigma
        self.k = k
        self.window_size = window_size
        self.conv_op = ConvolutionOperator()
        
    def compute_gradients(self, image: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], 
                                                                        npt.NDArray[np.float32]]:
        """Compute image gradients using Sobel operators"""
        sobel_x, sobel_y = KernelFactory.sobel()
        grad_x = self.conv_op.apply(image, sobel_x)
        grad_y = self.conv_op.apply(image, sobel_y)
        return grad_x, grad_y
        
    def compute_response(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute Harris response matrix"""
        grad_x, grad_y = self.compute_gradients(image)
        
        # Compute products of gradients
        Ixx = grad_x ** 2
        Ixy = grad_x * grad_y
        Iyy = grad_y ** 2
        
        # Apply Gaussian smoothing
        gaussian = KernelFactory.gaussian(self.window_size, self.sigma)
        Sxx = self.conv_op.apply(Ixx, gaussian)
        Sxy = self.conv_op.apply(Ixy, gaussian)
        Syy = self.conv_op.apply(Iyy, gaussian)
        
        # Compute Harris response
        det_M = (Sxx * Syy) - (Sxy ** 2)
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)
        
        return R
        
    def find_corners(self, R: npt.NDArray[np.float32], 
                    threshold: float = 0.01) -> List[Tuple[int, int]]:
        """Find corner points from Harris response matrix"""
        corners = []
        R_max = np.max(R)
        threshold_value = threshold * R_max
        offset = self.window_size // 2
        height, width = R.shape
        
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                if R[y, x] > threshold_value:
                    window = R[y-offset:y+offset+1, x-offset:x+offset+1]
                    if R[y, x] == np.max(window):
                        corners.append((y, x))
                        
        return corners

class Visualizer:
    """Handles visualization of detected corners"""
    
    @staticmethod
    def display_corners(image: npt.NDArray[np.float32], 
                       corners: List[Tuple[int, int]], 
                       title: str = "Detected Corners"):
        """Display image with detected corners"""
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='gray')
        
        if corners:
            y_coords, x_coords = zip(*corners)
            plt.scatter(x_coords, y_coords, color='red', s=10)
            
        plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    # Initialize detector
    detector = HarrisDetector(sigma=1.5, k=0.04, window_size=3)
    
    # Process images
    image_paths = [
        r'./2-1.jpg',
        r'./2-2.jpg'
    ]
    
    for idx, path in enumerate(image_paths, 1):
        try:
            # Load and process image
            image_data = ImageData.from_path(path)
            response = detector.compute_response(image_data.array)
            corners = detector.find_corners(response, threshold=0.01)
            
            # Visualize results
            Visualizer.display_corners(
                image_data.array, 
                corners, 
                f"Detected Corners - Image {idx}"
            )
            print(f"Processed image {idx}: Found {len(corners)} corners")
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
    
    print("Corner detection completed.")

if __name__ == "__main__":
    main()