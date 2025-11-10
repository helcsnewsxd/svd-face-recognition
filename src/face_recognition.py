import os
import logging
import argparse
import numpy as np
from typing import List, Tuple
from image import Image, read_images


class FaceRecognition:
    """
    Python SVD Face Recognition model class.

    Attributes
    ----------
    without_threshold: bool
        Specifies if model will use thresholds or not
    e0: float
        Threshold to know if image is a known face
    e1: float
        Threshold to know if image is a face
    k: int
        Compression level
    LOG: bool
        Logging flag
    shape: Tuple[int, int]
        Shape of images
    imgs: List[Tuple[npt.NDArray[np.float64], str]]
        List of tuples with compressed image and its tag
    M: int
        Total number of pixels in each image
    N: int
        Total number of images in training set
    fmean: npt.NDArray[np.float64]
        Vector that represents mean face
    A: npt.NDArray[np.float64]
        Matrix A used in SVD
    U: npt.NDArray[np.float64]
        Matrix U from SVD of A
    X: npt.NDArray[np.float64]
        Matrix that represents images in face space
    """

    def __init__(
        self,
        without_threshold: bool,
        e0: float | None,
        e1: float | None,
        k: int,
        LOG: bool,
        shape: Tuple[int, int],
    ) -> None:
        """
        FaceRecognition constructor.

        Parameters
        ----------
        without_threshold: bool
            Specifies if model will use thresholds or not
        e0: float
            Threshold to know if image is a known face
        e1: float
            Threshold to know if image is a face
        k: int
            Compression level
        LOG: bool
            Logging flag
        shape: Tuple[int, int]
            Shape of images

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If thresholds are not positive values, compression level k is not positive or
            if thresholds are not provided when without_threshold is False
        """
        self.without_threshold = without_threshold
        if not without_threshold:
            if e0 is None or e1 is None:
                raise ValueError("Thresholds must to have values")
            elif e0 <= 0 or e1 <= 0:
                raise ValueError("Threshold values must be positives")
            self.e0, self.e1 = e0, e1

        if k <= 0:
            raise ValueError("Compression value must be positive")
        self.k = k
        self.LOG = LOG
        self.shape = shape

        self.imgs = []
        self.M, self.N = shape[0] * shape[1], 0
        self.fmean = np.zeros((0,))
        self.A = np.zeros((0, 0))
        self.U = np.zeros((0, 0))
        self.X = np.zeros((0, 0))

    def train(self, images: List[str], tags: List[str]) -> None:
        """
        Trains the FaceRecognition model.

        Parameters
        ----------
        images: List[str]
            List of paths to images for training
        tags: List[str]
            List of face names for each image path

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If images list is empty or if images and tags lists have different sizes
        """
        if len(images) == 0:
            raise ValueError("Model can't train without training data")
        elif len(images) != len(tags):
            raise ValueError("Image paths list must have same size as tags list")

        logging.info("Starting model training...")

        self.imgs = list(
            zip(
                [
                    img.compressed_image.flatten()  # type: ignore[arg-type]
                    for img in read_images(images, self.k, self.LOG, self.shape)
                ],  # type: ignore[arg-type]
                tags,
            )
        )

        # Construct S
        self.N = len(self.imgs)
        S = np.column_stack([c for c, _ in self.imgs])

        # Get A
        self.fmean = np.mean(S, axis=1)
        self.A = S - self.fmean.reshape(-1, 1)

        # Get orthogonal U
        self.U, _, _ = np.linalg.svd(self.A, full_matrices=False)

        # Calculate image coordinates in face space
        self.X = self.U.T @ self.A

    def classify(self, src: str) -> Tuple[str, np.floating | float] | None:
        """
        Classifies a given image.

        Parameters
        ----------
        src: str
            Path to the image to be classified

        Returns
        -------
        (tag: str, distance: float) | None
            Returns a tuple with the tag of the recognized face and the distance value if the face is known,
            or None if the face is unknown or not a face.

        Raises
        ------
        RuntimeError:
            If the model has not been trained yet
        """
        if self.N == 0:
            raise RuntimeError("Model can't classify without previous training")

        logging.info("Starting model classification query")

        f = Image.from_file(src, self.k, self.shape).compressed_image.flatten()  # type: ignore[arg-type]
        x = self.U.T @ (f - self.fmean)

        # Check if it's a face (e1)
        fp = self.U @ x

        if not self.without_threshold:
            is_face = np.linalg.norm((f - self.fmean) - fp, ord=2)
            if is_face > self.e1:
                logging.info(f"{src} isn't a face. Value equal to {is_face}.")
                return None

        # Check if it's a known face (e0) and get it
        qnt = 0
        specific_face = -1
        idx = -1
        if not self.without_threshold:
            for j in range(self.N):
                eps = np.linalg.norm(x - self.X[:, j], ord=2)
                if eps <= self.e0:
                    qnt += 1
                    if idx < 0 or eps < specific_face:
                        specific_face = eps
                        idx = j
        else:
            qnt = 1
            for j in range(self.N):
                eps = np.linalg.norm(x - self.X[:, j], ord=2)
                if idx < 0 or eps < specific_face:
                    specific_face, idx = eps, j

        if qnt == 0:
            logging.info(f"{src} is an unknown face.")
            return None
        else:
            logging.info(
                f"{src} is a known face. "
                f"{qnt} proposed faces. "
                f"Closer image is {self.imgs[idx][1]} with eps = {specific_face}."
            )
            return self.imgs[idx][1], specific_face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition module using SVD")
    parser.add_argument("target", type=str, help="Target image to identify face")
    parser.add_argument(
        "--images",
        nargs="+",
        type=str,
        help="List of directory paths to images for training",
    )
    parser.add_argument(
        "--tags", nargs="+", type=str, help="Face names for each directory path"
    )
    parser.add_argument("-k", required=True, type=int, help="Compression level")
    parser.add_argument("--shape", nargs=2, type=int, help="Image size")
    parser.add_argument(
        "--without_threshold",
        action="store_true",
        help="If we want to run FaceRecognition without threshold (to get important values to calculate them)",
    )
    parser.add_argument(
        "-e0",
        type=float,
        help="Threshold to know if image is a known face",
    )
    parser.add_argument("-e1", type=float, help="Threshold to know if image is a face")
    parser.add_argument("--log", type=str, choices=["DEBUG", "INFO"], help="Log level")
    args = parser.parse_args()

    if len(args.images) != len(args.tags):
        raise parser.error(
            "Lists --images and --tags must to have saem quantity of elements."
        )
    elif args.without_threshold and (args.e0 or args.e1):
        raise parser.error(
            "If we specify without_threshold, then values e0, e1 won't be considered"
        )
    elif not args.without_threshold and (args.e0 is None and args.e1 is None):
        raise parser.error("Thresholds must be defined")

    logging.basicConfig(
        level=getattr(logging, args.log if (args.log is not None) else "WARNING"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    LOG = args.log is not None

    # To check
    images, tags = [], []
    for dir, tag in zip(args.images, args.tags):
        for file in os.listdir(dir):
            if os.path.isfile(f"{dir}/{file}"):
                images.append(f"{dir}/{file}")
                tags.append(tag)

    model = FaceRecognition(
        args.without_threshold, args.e0, args.e1, args.k, LOG, args.shape
    )
    model.train(images, tags)
    ans = model.classify(args.target)

    if ans is None:
        print("Face doesn't found")
    else:
        print(f"Image is face for {ans[0]} with distance {ans[1]}")
