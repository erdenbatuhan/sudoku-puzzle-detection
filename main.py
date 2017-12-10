""" Batuhan Erden S004345 Department of Computer Science """


import numpy as np
import glob
import cv2
import os
from matplotlib import pyplot as plt

import SudokuRecognizer as sr
from mnist_web import mnist


def print_confusion_matrix(confusion_matrix):
    print("\nConfusion matrix where rows are actual values and columns are predicted values..")

    print("", end="\t\t")
    for i in range(10):
        print("%d" % i, end="\t")
    print("\n--------------------------------------------------")

    for i in range(len(confusion_matrix)):
        print("%d | " % i, end="\t")
        for j in confusion_matrix[i]:
            print("%d" % j, end="\t")
        print()

    print()


# Set dataset path before start example  '/Home/sudoku_dataset-master' :
sudoku_dataset_dir = "./dataset"
MNIST_dataset_dir = "./MNIST_dataset"

# Load MNIST Dataset:
train_images, train_labels, test_images, test_labels = mnist(path=MNIST_dataset_dir)

# Apply PCA to MNIST
train_images, test_images, eigenvectors = sr.mnistPCA(train_images, test_images)

train_data = np.dot(train_images, eigenvectors)
sample_data = np.dot(test_images, eigenvectors)


num_test_images = len(test_images)

confusion_matrix = np.zeros((10, 10), dtype=np.uint8)
accuracy = 0.

print("Testing accuracy on MNIST dataset, this might take a while..")
for i in range(num_test_images):
    nearest_index, value = sr.get_nearest_euclidean_distance(sample_data[i], train_data)

    train_label = sr.get_label(train_labels[nearest_index])
    sample_label = sr.get_label(test_labels[i])

    if train_label == sample_label:
        accuracy += 1 / len(test_images)

    confusion_matrix[sample_label, train_label] += 1

print_confusion_matrix(confusion_matrix)
print("MNIST dataset performance: {:.2f}%".format(accuracy * 100))


image_dirs = sudoku_dataset_dir + "/images/image*.jpg"
data_dirs = sudoku_dataset_dir + "/images/image*.dat"
IMAGE_DIRS = glob.glob(image_dirs)
DATA_DIRS = glob.glob(data_dirs)
len(IMAGE_DIRS)


# Accumulate accuracy for average accuracy calculation.
confusion_matrix = np.zeros((10, 10), dtype=np.uint8)
cumulativeAcc = 0

# Loop over all images
print("-----------")
print("Testing accuracy on Sudoku dataset, this might take a while..")
for img_dir, data_dir in zip(IMAGE_DIRS, DATA_DIRS):
    image_name = os.path.basename(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
    img = cv2.imread(img_dir)

    # Uncomment this section if you would like to see resulting bounding boxes.
    # bounding_boxes = sr.detectSudoku(img)
    # cv2.rectangle(img, bounding_boxes[0][0], bounding_boxes[0][1], (0, 0, 255), 2)
    # cv2.imshow(image_name, img)
    # cv2.waitKey()

    # Recognize digits in sudoku puzzle:
    try:
        sudokuArray = sr.RecognizeSudoku(img, train_images, train_labels)
    except ValueError:  # The case where sudoku cannot be found in the image
        sudokuArray = np.zeros((9, 9), dtype=np.uint8)

    # Evaluate Result for current image:
    detectionAccuracyArray = data == sudokuArray
    accPercentage = np.sum(detectionAccuracyArray)/detectionAccuracyArray.size
    cumulativeAcc = cumulativeAcc + accPercentage

    straight_data = np.reshape(data, (81, ))
    straight_sudokuArray = np.reshape(sudokuArray, (81, ))

    for i in range(81):
        actual = straight_data[i]
        predicted = straight_sudokuArray[i]

        confusion_matrix[actual, predicted] += 1

    print(image_name + " accuracy : " + str(accPercentage * 100) + "%")

# Calculate confusion matrix, false postivies/negatives for Sudoku dataset here.
print_confusion_matrix(confusion_matrix)

# Average accuracy over all images in the dataset:
averageAcc = cumulativeAcc/len(IMAGE_DIRS)
print("Sudoku dataset performance: " + str(averageAcc * 100) + "%")

