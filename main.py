import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data():
    digits = load_digits()
    images = digits.images
    labels = digits.target
    return images, labels


def prepare_data(images):
    # model needs a flat list of numbers for each image, not 8x8
    n = len(images)
    x = images.reshape(n, -1)

    # pixels are 0 to 16, make them 0 to 1
    x = x / 16.0
    return x


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

x = prepare_data(images)
print()
print("Data shape after flatten:", x.shape)
print("Min and max value:", x.min(), x.max())
print("First row of x:", x[0][:10])

# 80% for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=1)
print()
print("Train size:", len(x_train))
print("Test size:", len(x_test))

# make sure test set has all the digits
print("Digits in test set:", np.unique(y_test))

# first model, knn is simple so start with it
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
print()
print("Model trained with", model.n_samples_fit_, "images")

# predict the test images
predictions = model.predict(x_test)
print()
print("Predicted:", predictions[:10])
print("Real     :", y_test[:10])

# show the first test image and what the model thinks
first = x_test[0].reshape(8, 8) * 16
show_digit(first)
print("Prediction:", predictions[0])

# how many did the model get right
correct = np.sum(predictions == y_test)
print()
print("Correct:", correct, "out of", len(y_test))

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")

# find the wrong ones
wrong = np.where(predictions != y_test)[0]
print("Wrong indexes:", wrong)

print()
print("Example image, label is", labels[0])
show_digit(images[0])
