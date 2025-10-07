import random
from .thresholding import otsu_threshold
from .blurring import gaussian_blur, median_blur


def preprocess_image(x, blur: bool = True, threshold: bool = True):
    """
    Preprocesses the image by optionally applying a random blur and/or thresholding.

    @Usage:
        If `blur=True`, applies either Gaussian or median blurring (chosen at random) to reduce noise.
        If `threshold=True`, applies Otsu's thresholding to binarize the image.
        You can enable either step independently or both in sequence.

    @Parameters:
    x : np.ndarray
        Input image array.
    blur : bool, optional (default=False)
        Whether to apply a random blur (Gaussian or Median).
    threshold : bool, optional (default=False)
        Whether to apply Otsu's thresholding.

    @Returns:
    np.ndarray : Preprocessed image after the selected steps.
    """
    # Randomly select one of the two blurring methods (only if blur is enabled)
    if blur:
        blur_method = random.choice([gaussian_blur, median_blur])
        x = blur_method(x)

    # Apply Otsu's thresholding (only if threshold is enabled)
    if threshold:
        x = otsu_threshold(x)

    return x
