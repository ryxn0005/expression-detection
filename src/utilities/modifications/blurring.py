import cv2

def gaussian_blur(x, kernel_size=3):
    """
    Applies Gaussian blur to the input image.

    @Usage:
        Smooths the image by applying a Gaussian kernel, which reduces noise and detail.

    @Parameters:
    x : np.ndarray
        Input image array.
    kernel_size : int, optional, default=3
        Size of the kernel to use for blurring. Must be an odd number.

    @Returns:
    np.ndarray : Blurred image.
    """
    return cv2.GaussianBlur(x, (kernel_size, kernel_size), 0)


def median_blur(x, kernel_size=3):
    """
    Applies median blur to the input image.

    @Usage:
        Reduces noise by replacing each pixel's value with the median of neighboring pixels.

    @Parameters:
    x : np.ndarray
        Input image array.
    kernel_size : int, optional, default=3
        Size of the kernel to use for median blurring. Must be an odd number.

    @Returns:
    np.ndarray : Blurred image.
    """
    return cv2.medianBlur(x, kernel_size)
