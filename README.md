# Singular Value Decomposition Applied To Digital Image Processing

This project demonstrates the application of Singular Value Decomposition (SVD) in digital image processing. It includes functions to perform SVD on images, reconstruct images from their SVD components, and visualize the results.

But image compression isn't the main focus here; instead, we explore how SVD can be used for face recognition.
This idea was taken from the notes [Singular Value Decomposition Applied To Digital Image Processing](https://www.math.cuhk.edu.hk/~lmlui/CaoSVDintro.pdf) of Lijie Cao.

In this repository, we provide Python modules that implement SVD for image processing tasks, along with example scripts to demonstrate their usage.
Also, there are two datasets of face images included for testing and demonstration purposes (Celebrity Faces and Yale B Face Database), with a Jupyter notebook showcasing face recognition using SVD and visualization of the metric results.

Finally, we include a report in PDF format that details the methodology, experiments, and findings related to the application of SVD in face recognition.

## Contents

<div align="center">

| Content                                                          | Description                                                                                                                   |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [Celebrity Faces Dataset](./data/celebrities/)                   | A dataset of celebrity face images for testing SVD-based face recognition.                                                    |
| [Yale B Face Database](./data/yale/)                             | A dataset of face images under different lighting conditions for testing SVD-based face recognition.                          |
| [image.py](./src/image.py)                                       | Python module implementing SVD functions for image preprocessing (normalization, reshape, gray-scale, compression).           |
| [face_recognition.py](./src/face_recognition.py)                 | Python module implementing SVD-based face recognition functions (training, classify).                                         |
| [get_data.py](./src/get_data.py)                                 | Python script to execute face recognition experiments using the provided datasets.                                            |
| [distances.txt](./metrics/distances.txt)                         | Results file containing distance metrics from face recognition experiments without thresholding, for Yale B Face Database.    |
| [celebrities_distances.txt](./metrics/celebrities_distances.txt) | Results file containing distance metrics from face recognition experiments without thresholding, for Celebrity Faces Dataset. |
| [analysis.ipynb](./metrics/analysis.ipynb)                       | Jupyter notebook for visualizing and analyzing the distance metrics from face recognition experiments.                        |
| [report.pdf](./report/report.pdf)                                | PDF report detailing the methodology, experiments, and findings related to SVD in face recognition.                           |

</div>
