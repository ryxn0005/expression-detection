import random
from .thresholding import otsu_threshold
from .blurring import gaussian_blur, median_blur


def preprocess_image(x):
    """
    Preprocesses the image by applying a random blur and then thresholding.

    @Usage:
        First, applies either Gaussian or median blurring, chosen at random, to reduce noise.
        Then, applies Otsu's thresholding to binarize the image.

    @Parameters:
    x : np.ndarray
        Input image array.

    @Returns:
    np.ndarray : Preprocessed image after blurring and thresholding.
    """
    # Randomly select one of the two blurring methods
    blur_method = random.choice([gaussian_blur, median_blur])
    x_blurred = blur_method(x)

    # Apply Otsu's thresholding after blurring
    x_thresholded = otsu_threshold(x_blurred)

    return x_thresholded
