import cv2
import numpy as np

def otsu_threshold(x):
    """
    Applies Otsu's thresholding to binarize the image.

    @Usage:
        This function first converts the image to an 8-bit format, applies Otsu's thresholding
        for automatic thresholding, and then converts the result back to a float format.

    @Parameters:
    x : np.ndarray
        Input image array. Expects a grayscale image in float32 format.

    @Returns:
    np.ndarray : Thresholded (binarized) image in float32 format.
    """
    # Temporarily convert to 8-bit for OpenCV processing
    if x.dtype != np.uint8:
        x_uint8 = (x * 255).astype(np.uint8)
    else:
        x_uint8 = x

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(x_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to float32 for compatibility
    thresholded_float = thresholded.astype(np.float32) / 255.0
    return thresholded_float
