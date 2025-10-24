import os
import cv2
import logging
import argparse
import numpy as np
import numpy.typing as npt
from math import log10
from tqdm import tqdm
from typing import List


class Image:
    """
    Python Image class for file-readable images and preprocessing (aiming SVD Face Recognition)

    Attributes
    ----------
    image: npt.NDArray[np.uint8]
        Original image read from file

    gray_image: npt.NDArray[np.uint8] | None
        Original image with GrayScale filter

    compressed_image: npt.NDArray[np.uint8] | None
        Compressed GrayScale image

    A: npt.NDArray[np.float64] | None
        Matrix that represents compressed grayscale image

    U: npt.NDArray[np.float64] | None
        Matrix U for A's SVD decomposition

    S: npt.NDArray[np.float64] | None
        Matrix S for A's SVD decomposition

    Vh: npt.NDArray[np.float64] | None
        Matrix Vh for A's SVD decomposition

    k: int
        Compression level
    """

    def __init__(self, image: npt.NDArray[np.uint8], k: int) -> None:
        """
        Image constructor with preprocessing applied to it.

        Parameters
        ----------
        image: npt.NDArray[np.uint8]
            Original image
        k: int
            Compression level

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If compression level k exceeds image matrix rank
        """

        self.image = image
        self.gray_image = None
        self.compressed_image = None

        self.A = None
        self.U, self.S, self.Vh = None, None, None
        self.k = k
        self.preprocessing(k)

    @classmethod
    def from_file(cls, file: str, k: int):
        """
        Image constructor reading image from file.

        Parameters
        ----------
        file: str
            Image file
        k: int
            Compression level

        Returns
        -------
        Image class

        Raises
        ------
        FileNotFoundError:
            If file doesn't exists
        IsADirectoryError:
            If file-string doesn't represent a file
        ValueError:
            If compression level k exceeds image matrix rank
        """

        if not os.path.exists(file):
            logging.error(f"Image CLS creation -> File {file} doesn't exists.")
            raise FileNotFoundError(f"File {file} doesn't exists.")
        elif not os.path.isfile(file):
            logging.error(
                f"Image CLS creation -> File {file} isn't a file. It's a directory."
            )
            raise IsADirectoryError(f"File {file} isn't a file. It's a directory.")

        logging.debug(f"Reading image from {file}")

        image = cv2.imread(file)

        if image is None:
            logging.error("Reading failure.")
            raise ValueError(f"Failed to read image from {file}.")

        return cls(image, k)  # type: ignore

    def preprocessing(self, k: int) -> None:
        """
        Preprocessing image with compression level k.
        Applies GrayScale filter and image compression using SVD.

        Parameters
        ----------
        k: int
            Compression level

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If compression level k exceeds image matrix rank
        """

        m, n = self.image.shape[0], self.image.shape[1]
        if k > n:
            raise ValueError(f"Invalid compression constant: {k} > {n}.")

        logging.debug("Image preprocessing.")
        self.k = k

        # BGR image to GRAY_SCALE
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Compress image
        U, S, Vh = np.linalg.svd(self.gray_image)
        self.U, self.S, self.Vh = (
            U[:, :k],
            S[:k],
            Vh[:k, :],
        )
        self.A = self.U @ np.diag(self.S) @ self.Vh
        self.compressed_image = cv2.normalize(
            self.A,
            None,  # type: ignore[arg-type]
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        # Compression metrics
        # PSNR: > 40dB indistinguishable
        #       [30-40]dB good
        #       < 25dB bad

        Cr = m * n / (k * (m + n + 1))
        MSE = np.mean((self.gray_image - self.A) ** 2)
        PSNR = 10 * log10(255**2 / MSE)
        logging.debug(
            f"Compression ratio: {Cr}\t|\tCompression MSE: {MSE}\t|\tCompression PSNR: {PSNR}"
        )
        if PSNR <= 25:
            logging.warning(
                f"Compression with low-quality obtained for k = {self.k} | PSNR = {PSNR}"
            )

        logging.debug("Success image preprocessing.")

    @property
    def show(self) -> None:
        """
        Show image and compare it with compression one.
        """

        if self.compressed_image is not None and self.gray_image is not None:
            cv2.imshow(
                f"Grayscale: Original vs. {self.k}-th compression",
                np.hstack((self.gray_image, self.compressed_image)),
            )
        else:
            cv2.imshow("Original image", self.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def read_images_from_directory(directory: str, k: int, LOG: bool) -> List[Image]:
    """
    Read images from directory and return a list of Image classes with
    preprocessed images.

    Parameters
    ----------
    directory: str
        Directory that contains all image files
    k: int
        Compression level
    LOG: bool
        If we want to see TQDM bar

    Returns
    -------
    List[Image]:
        List with all Image classes for each one in the directory

    Raises
    ------
    NotADirectoryError:
        If directory is invalid
    ValueError:
        If compression level k exceeds image matrix rank for some image into the directory
    """

    if not os.path.exists(directory):
        raise NotADirectoryError(f"Directory {directory} not found.")
    elif not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory {directory} isn't a directory.")

    image_list = []

    image_files = [
        f"{directory}/{file}"
        for file in os.listdir(directory)
        if os.path.isfile(f"{directory}/{file}")
    ]
    pbar = (
        tqdm(
            total=len(image_files), desc="Image reading and preprocessing", unit="Image"
        )
        if LOG
        else None
    )
    for file in image_files:
        image_list.append(Image.from_file(file, k))
        if pbar:
            pbar.update(1)

    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing module")
    parser.add_argument("directory", type=str, help="Directory with images to read")
    parser.add_argument("-k", required=True, type=int, help="Compression level")
    parser.add_argument("--log", type=str, choices=["DEBUG", "INFO"], help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log if (args.log is not None) else "WARNING"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # To check
    images = read_images_from_directory(
        args.directory, k=args.k, LOG=(args.log is not None)
    )
    for i in images:
        i.show
