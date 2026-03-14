# Brief description
This gesture classifier is the result of the first successful train|validation|test cycle i`ve ever made. For first (2D-conv) versions, see deprecated/. Its first version was also a part of my bachelors graduation work presented in deprecated/bachelors_diploma.pdf file - further descriptions can be found there.

The second version is trained for all the alphabet including dynamic gestures, leverages completely different architecture - Expand3D (X3D-M) - with kinetics-400 backbone.
For further descriptions, see Abstract.pdf.

The dataset can be found at: https://doi.org/10.5281/zenodo.17618405 (frame_data.zip)

# Illustrations:

1) training set (frame examples):
<img width="975" height="394" alt="image" src="https://github.com/user-attachments/assets/bc3def19-ef71-4712-bcba-5e9622d1d3d9" />

2) validation set (frame examples):
<img width="975" height="392" alt="image" src="https://github.com/user-attachments/assets/3b153ed8-d831-4124-8a72-7b4067e1c180" />

3) testing set (frame examples):
<img width="975" height="392" alt="image" src="https://github.com/user-attachments/assets/af6f6554-0106-4c32-bf96-f567687779fe" />

# Test set classification reports:

<img width="864" height="710" alt="image" src="https://github.com/user-attachments/assets/c1549c29-efa5-4c6f-98cc-86787e6e8aca" />

SK-learn classification report:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| а | 1.00 | 0.94 | 0.97 | 16 |
| б | 0.93 | 0.88 | 0.90 | 16 |
| в | 1.00 | 1.00 | 1.00 | 16 |
| г | 1.00 | 0.88 | 0.93 | 16 |
| ґ | 1.00 | 1.00 | 1.00 | 16 |
| д | 1.00 | 0.75 | 0.86 | 16 |
| е | 0.84 | 1.00 | 0.91 | 16 |
| є | 1.00 | 1.00 | 1.00 | 16 |
| ж | 0.93 | 0.88 | 0.90 | 16 |
| з | 1.00 | 0.81 | 0.90 | 16 |
| и | 0.93 | 0.88 | 0.90 | 16 |
| і | 1.00 | 0.94 | 0.97 | 16 |
| ї | 0.88 | 0.94 | 0.91 | 16 |
| й | 0.76 | 0.81 | 0.79 | 16 |
| к | 0.84 | 1.00 | 0.91 | 16 |
| л | 1.00 | 0.75 | 0.86 | 16 |
| м | 0.64 | 0.56 | 0.60 | 16 |
| н | 1.00 | 1.00 | 1.00 | 16 |
| о | 0.76 | 1.00 | 0.86 | 16 |
| п | 1.00 | 1.00 | 1.00 | 16 |
| р | 0.88 | 0.94 | 0.91 | 16 |
| с | 1.00 | 1.00 | 1.00 | 16 |
| т | 0.78 | 0.88 | 0.82 | 16 |
| у | 0.80 | 1.00 | 0.89 | 16 |
| ф | 0.80 | 1.00 | 0.89 | 16 |
| х | 1.00 | 1.00 | 1.00 | 16 |
| ц | 1.00 | 0.81 | 0.90 | 16 |
| ч | 0.82 | 0.88 | 0.85 | 16 |
| ш | 0.75 | 0.56 | 0.64 | 16 |
| щ | 0.89 | 1.00 | 0.94 | 16 |
| ь | 1.00 | 0.94 | 0.97 | 16 |
| ю | 0.94 | 1.00 | 0.97 | 16 |
| я | 1.00 | 1.00 | 1.00 | 16 |
|-------|-----------|--------|----------|---------|
| **accuracy** | - | - | 0.91 | 528 |
| **macro avg** | 0.92 | 0.91 | 0.91 | 528 |
| **weighted avg** | 0.92 | 0.91 | 0.91 | 528 |

