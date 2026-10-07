import numpy as np
from sklearn.datasets import load_digits


def load_data():
    digits = load_digits()
    images = digits.images
    labels = digits.target
    return images, labels


def show_digit(image):
    # print one 8x8 image as text
    for row in image:
        line = ""
        for pixel in row:
            if pixel > 8:
                line += "# "
            elif pixel > 0:
                line += ". "
            else:
                line += "  "
        print(line)


images, labels = load_data()

print("Handwriting Recognition")
print("-----------------------")
print("Number of images:", len(images))
print("Image shape:", images[0].shape)
print("Labels:", np.unique(labels))

# check how many images we have for each digit
for d in range(10):
    print(d, ":", np.sum(labels == d))

print()
print("Example image, label is", labels[0])
show_digit(images[0])
