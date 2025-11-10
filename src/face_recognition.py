import os
import logging
import argparse
import numpy as np
from typing import List, Tuple
from image import Image, read_images


class FaceRecognition:
    def __init__(
        self,
        e0: float,
        e1: float,
        k: int,
        LOG: bool,
        shape: Tuple[int, int],
    ) -> None:
        if e0 <= 0 or e1 <= 0 or k <= 0:
            raise ValueError("Threshold and compression values must be positives")

        self.e0, self.e1 = e0, e1
        self.k = k
        self.LOG = LOG
        self.shape = shape

        self.imgs = []
        self.M, self.N = shape[0] * shape[1], 0
        self.fmean = np.ndarray((0,))
        self.A, self.U, self.X = (
            np.ndarray((0, 0)),
            np.ndarray((0, 0)),
            np.ndarray((0, 0)),
        )

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
        self.U = np.linalg.svd(self.A, full_matrices=False).U

        # Calculate image coordinates in face space
        self.X = np.column_stack([self.U.T @ self.A[:, j] for j in range(self.N)])

    def classify(self, src: str) -> Tuple[str, np.floating | float] | None:
        if self.N == 0:
            raise RuntimeError("Model can't classify without previous training")

        logging.info("Starting model classification query")

        f = Image.from_file(src, self.k, self.shape).compressed_image.flatten()  # type: ignore[arg-type]
        x = self.U.T @ (f - self.fmean)

        # Check if it's a face (e1)
        fp = self.U @ x
        is_face = np.linalg.norm((f - self.fmean) - fp, ord=2)

        if is_face > self.e1:
            logging.info(f"{src} isn't a face. Value equal to {is_face}.")
            return None

        # Check if it's a known face (e0) and get it
        qnt = 0
        specific_face = -1
        idx = -1
        for j in range(self.N):
            eps = np.linalg.norm(x - self.X[:, j], ord=2)
            if eps <= self.e0:
                qnt += 1
                if idx < 0 or eps < specific_face:
                    specific_face = eps
                    idx = j

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
        "-e0",
        required=True,
        type=float,
        help="Threshold to know if image is a known face",
    )
    parser.add_argument(
        "-e1", required=True, type=float, help="Threshold to know if image is a face"
    )
    parser.add_argument("--log", type=str, choices=["DEBUG", "INFO"], help="Log level")
    args = parser.parse_args()

    if len(args.images) != len(args.tags):
        raise parser.error(
            "Lists --images and --tags must to have saem quantity of elements."
        )

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

    model = FaceRecognition(args.e0, args.e1, args.k, LOG, args.shape)
    model.train(images, tags)
    ans = model.classify(args.target)

    if ans is None:
        print("Face doesn't found")
    else:
        print(f"Image is face for {ans[0]}")
