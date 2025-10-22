import os
import argparse
import logging as log
import numpy as np
import numpy.typing as npt
from PIL import Image
from typing import Tuple, List


# def preprocess_image(image_path: str, shape: Tuple[int, int]) -> npt.NDArray[np.float64]:
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"No se pudo cargar la imagen: {image_path}")

#     # Redimensionar
#     image = cv2.resize(image, shape)

#     # Normalizar a [0, 1]
#     image = image.astype(np.float64) / 255.0

#     # Convertir a vector 1D
#     return image.flatten()

SHAPE = (50, 50)


def get_matrix_values(file_instances: List[str]) -> npt.NDArray[np.float64]:
    global SHAPE
    images = list(map(lambda f: Image.open(f).convert('L').resize(SHAPE), file_instances))
    arrays = list(map(lambda i: np.asarray(i, dtype=np.float64).flatten(), images))
    return np.asarray(arrays, dtype=np.float64)


def create_new_training(input: str, output: str):
    if not os.path.exists(input):
        raise FileNotFoundError("Input file/directory doesn't exists")

    file_instances = None
    if os.path.isfile(input):
        file_instances = [input]
    else:
        file_instances = list(filter(lambda x: os.path.isfile(x), map(lambda y: f"{input}/{y}", os.listdir(input))))

    S = get_matrix_values(file_instances)
    fmean = np.mean(S, axis=0)
    A = np.array(list(map(lambda f: f-fmean, S)), dtype=np.float64)
    U, S, V = np.linalg.svd(A)


def main(generate_training: Tuple[str, str]):
    if generate_training is not None:
        create_new_training(generate_training[0], generate_training[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVD Face Recognition")

    parser.add_argument("--debug", action="store_true", help="Debugging mode")
    parser.add_argument(
        "--generate_training", nargs=2, default=None,
        help="Create new training instance from scratch (first argument as input and second as output)")

    args = parser.parse_args()

    if args.debug:
        log.basicConfig(
            level=log.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    main(args.generate_training)
