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


def train_model(x_train, y_train, k=3, weights="uniform"):
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    model.fit(x_train, y_train)
    return model


def get_accuracy(model, x_test, y_test):
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)


def find_best_k(x_train, y_train, x_test, y_test):
    best_k = 3
    best_acc = 0
    for k in [1, 3, 5, 7, 9, 15]:
        model = train_model(x_train, y_train, k)
        acc = get_accuracy(model, x_test, y_test)
        print("k =", k, "->", round(acc * 100, 2), "%")
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k


def show_example(model, x_test, y_test, i):
    img = x_test[i].reshape(8, 8) * 16
    prediction = model.predict(x_test[i].reshape(1, -1))[0]
    print()
    show_digit(img)
    print("Prediction:", prediction)
    print("Real digit:", y_test[i])


print("Handwriting Recognition")
print("-----------------------")

images, labels = load_data()
print("Number of images:", len(images))
print("Image shape:", images[0].shape)

# check how many images we have for each digit
for d in range(10):
    print(d, ":", np.sum(labels == d))

x = prepare_data(images)
print()
print("Data shape after flatten:", x.shape)
print("Min and max value:", x.min(), x.max())

# 80% for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=1)
print("Train size:", len(x_train))
print("Test size:", len(x_test))

# first model, knn is simple so start with it
model = train_model(x_train, y_train)
predictions = model.predict(x_test)
print()
print("Predicted:", predictions[:10])
print("Real     :", y_test[:10])

correct = np.sum(predictions == y_test)
print("Correct:", correct, "out of", len(y_test))
print("Accuracy:", round(get_accuracy(model, x_test, y_test) * 100, 2), "%")

# try different settings
print()
print("Testing different settings")
best_k = find_best_k(x_train, y_train, x_test, y_test)
print("Best k:", best_k)

# same thing but closer neighbors count more
for k in [3, 5, 9]:
    m = train_model(x_train, y_train, k, "distance")
    acc = get_accuracy(m, x_test, y_test)
    print("k =", k, "distance weights ->", round(acc * 100, 2), "%")

# final model with the best k
best_model = train_model(x_train, y_train, best_k)
best_predictions = best_model.predict(x_test)

print()
print("Example predictions (k =", best_k, ")")
for i in range(5):
    show_example(best_model, x_test, y_test, i)

# also look at the ones it got wrong
wrong = np.where(best_predictions != y_test)[0]
print()
print("Wrong with best model:", len(wrong))
for i in wrong:
    show_example(best_model, x_test, y_test, i)
