# Handwriting Recognition

A small machine learning project that recognizes handwritten digits (0 to 9). It loads a dataset of digit images, trains a simple scikit-learn model, and then predicts which digit is in each test image. Everything runs in the terminal.

## How it works

1. Load the digits dataset from scikit-learn (1797 images, each 8x8 pixels)
2. Flatten every image into 64 numbers and scale the pixel values to 0-1
3. Split the data into 80% training and 20% test
4. Train a K-Nearest Neighbors classifier
5. Predict the test images and calculate the accuracy
6. Try different `k` values (and distance weights) and pick the best one
7. Show some example predictions in the terminal, including the ones the model got wrong

## Example output

Each digit is drawn as text, followed by the prediction:

```
    . # #       
    # # # .     
    . # # .     
    # # #       
    # # #       
    # # #       
    # # #       
    . # # .     
Prediction: 1
Real digit: 1
```

With `k = 7` the model gets about 99.7% accuracy on the test set (359 out of 360 images).

## Tech

- Python
- NumPy
- scikit-learn

## Project structure

```
.
├── main.py
└── .gitignore
```

## Running locally

You need Python 3 and two packages:

```bash
pip install numpy scikit-learn
```

Then clone the repo and run it:

```bash
git clone https://github.com/sara-razavi/handwriting-recognition.git
cd handwriting-recognition
python main.py
```


## Author

**Sara Razavi**: [GitHub](https://github.com/sara-razavi)
