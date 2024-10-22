import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def create_gaussian_kernel(k, s):
    """
    Create a 2D Gaussian kernel.
    """
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * s**2))
    return kernel / np.sum(kernel)

def apply_convolution(image, kernel):
    """
    Apply convolution to the image using the given kernel.
    """
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    output = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            output[i,j] = np.sum(image[i:i+m, j:j+n]*kernel)
    return output

def load_image(path):
    """
    Load and preprocess an image.
    """
    image = np.array(Image.open(path).convert('L'))
    return image.astype(np.float32) / 255.0