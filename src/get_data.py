import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from face_recognition import FaceRecognition


def main(
    directory: str, shape: tuple[int, int], k: int, output: str | None, seed: int | None
):
    if not os.path.isdir(directory):
        raise NotADirectoryError(
            f"Directory {directory} not found or isn't a directory."
        )

    f = open(output, "w") if output is not None else None

    # Get all image paths and labels
    X, y = [], []
    qnt_per_person, classes = 0, 0
    for label in os.listdir(directory):
        if os.path.isdir(f"{directory}/{label}"):
            classes += 1
            act = 0
            for file in os.listdir(f"{directory}/{label}"):
                if os.path.isfile(f"{directory}/{label}/{file}"):
                    X.append(f"{directory}/{label}/{file}")
                    y.append(label)
                    act += 1

            if qnt_per_person == 0:
                qnt_per_person = act
            else:
                assert qnt_per_person == act, (
                    "All persons must have the same number of images."
                )

    X, y = np.array(X), np.array(y)
    print(
        f"Found {classes} classes with {qnt_per_person} images each. Total images: {len(X)}"
    )
    if f is not None:
        f.write(
            f"Classes found: {classes}\n"
            f"Images per class: {qnt_per_person}\n"
            f"Total images: {len(X)}\n\n"
        )

    # Divide train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    print(f"Train set: {len(X_train)} images. Test set: {len(X_test)} images.")

    # Train face recognition model
    print("Starting training...")
    model = FaceRecognition(
        without_threshold=True, e0=None, e1=None, k=k, LOG=False, shape=shape
    )
    model.train(X_train, y_train)
    print("Training finished.")

    # Get distances for each face image class
    print("Getting distances for test set...")
    pbar = (
        tqdm(total=len(X_test), desc="Getting distances", unit="Image")
        if f is not None
        else None
    )
    for i, img in enumerate(X_test):
        res = model.get_distances(img)
        if f is not None:
            f.write(f"=== Distances for image with label {y_test[i]} ===\n")
            for key, value in res:
                f.write(f"- {key}: {value}\n")
            f.write("\n")

            assert pbar is not None
            pbar.update(1)
        else:
            print(f"=== Distances for image with label {y_test[i]} ===")
            for key, value in res:
                print(f"- {key}: {value}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition using SVD")
    parser.add_argument("directory", type=str, help="Directory with images to read")
    parser.add_argument(
        "--shape", nargs=2, type=int, help="Image size", default=(100, 100)
    )
    parser.add_argument("-k", required=True, type=int, help="Compression level")
    parser.add_argument("--output", type=str, help="Output file for distances")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    main(args.directory, args.shape, args.k, args.output, args.seed)
