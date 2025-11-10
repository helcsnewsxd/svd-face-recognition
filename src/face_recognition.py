import os
import logging
import argparse
import numpy as np
from typing import List, Tuple
from image import Image, read_images


class FaceRecognition:
    def __init__(
        self,
        without_threshold: bool,
        e0: float | None,
        e1: float | None,
        k: int,
        LOG: bool,
        shape: Tuple[int, int],
    ) -> None:
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
