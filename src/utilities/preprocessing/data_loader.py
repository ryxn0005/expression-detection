import os, glob
import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple, Dict, Union


def load_data(
    dataset_path: str,
    image_size: int = 48,
    preprocess_fn=None,
    exts: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    shuffle_seed: int = 42,
    export_label: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, int]]
]:
    """
    Load a grayscale image dataset from class subfolders into (X, y), with labels inferred
    from folder names and mapped to consecutive integers in sorted order.

    @Usage:
        Place one folder per class inside `dataset_path`:
            dataset_path/
                class_a/  *.jpg|*.png|...
                class_b/  *.jpg|*.png|...
                ...
        Optionally pass a preprocessing function that takes a (H, W) grayscale array
        and returns a processed array. Set `include_label_map=True` if you also want
        the {class_name: index} mapping returned.

    @Parameters:
        dataset_path : str
            Path to the directory containing class subfolders (e.g., "./fane_data").
        image_size : int, optional, default=64
            Target height/width for resizing each image (square).
        preprocess_fn : callable or None, optional, default=None
            Function applied to each grayscale image after resizing. Signature:
            `preprocess_fn(img: np.ndarray) -> np.ndarray`.
        exts : tuple[str], optional, default=(".jpg", ".jpeg", ".png", ".bmp")
            File extensions to include when scanning for images.
        shuffle_seed : int, optional, default=42
            Seed for reproducible shuffling.
        include_label_map : bool, optional, default=False
            If True, also return the {class_name: index} mapping.

    @Returns:
        tuple :
            If include_label_map is False:
                (X, y, categories)
            If include_label_map is True:
                (X, y, categories, label_map)

            X : np.ndarray
                Shape (N, image_size, image_size, 1), dtype float32.
            y : np.ndarray
                Integer class indices, shape (N,), dtype int64.
            categories : list[str]
                Sorted list of class names (index order matches y).
            label_map : dict[str, int]  [only if include_label_map=True]
                Mapping {class_name: index} reflecting the sorted order.
    """
    # collect class folders (sorted for stable indices)
    categories = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    label_map = {c: i for i, c in enumerate(categories)}
    if export_label:
        print("categories:", categories)
        print("label_map:", label_map)

    data = []  # list of (img, label_idx)

    for category in categories:
        full_category_path = os.path.join(dataset_path, category)
        label_idx = label_map[category]

        # recursively gather files
        files = []
        for root, _, _ in os.walk(full_category_path):
            for ext in exts:
                files.extend(glob.glob(os.path.join(root, f"*{ext}")))

        for img_path in tqdm(files, desc=f"Processing {category}", leave=False):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Unable to load image {img_path}. Skipping.")
                    continue

                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

                # optional custom preprocessing
                if preprocess_fn is not None:
                    img = preprocess_fn(img, blur=False, threshold=False)

                data.append((img, label_idx))
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue

    # single shuffle at the end (reproducible)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(data)

    # split into arrays
    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.int64)

    X = X / 256

    # ensure channel dim for Keras
    if X.ndim == 3:
        X = X[..., None]  # (N, H, W, 1)

    if export_label:
        return X, y, label_map
    else:
        return X, y

