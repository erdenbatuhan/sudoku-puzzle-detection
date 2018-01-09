""" Batuhan Erden S004345 Department of Computer Science """


import numpy as np
import pickle
import glob
import cv2
import os
from matplotlib import pyplot as plt

import SudokuRecognizer as sr
from mnist_web import mnist
from sklearn import svm


# Set dataset paths before start. For example: '/Home/sudoku_dataset-master'
SUDOKU_DATASET_DIR = "./dataset"
MNIST_DATASET_DIR = "./MNIST_dataset"

IMAGE_DIRS = SUDOKU_DATASET_DIR + "/images/image*.jpg"
DATA_DIRS = SUDOKU_DATASET_DIR + "/images/image*.dat"


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


def load_mnist_dataset():
    print("Console: Loading MNIST dataset..")
    train_images, train_labels, test_images, test_labels = mnist(path=MNIST_DATASET_DIR)
    train_labels, test_labels = sr.convert_labels(train_labels), sr.convert_labels(test_labels)

    print("Console: MNIST dataset loaded.")
    return train_images, train_labels, test_images, test_labels


def apply_pca_to_mnist(train_images):
    print("Console: Applying PCA on MNIST dataset..")
    eigenvectors = sr.apply_pca(train_images)  # Apply PCA to train images

    train_images = (train_images - np.mean(train_images, axis=0))  # Subtract mean
    train_features = np.dot(train_images, eigenvectors)  # Dot product

    print("Console: PCA applied on MNIST dataset.")
    return train_features, eigenvectors


def build_classifier(train_features, train_labels, test_features, test_labels):
    try:
        with open("classifier.pkl", "rb") as reader:
            print("Console: A pre-saved classifier loaded.")
            classifier = pickle.load(reader)
    except OSError:
        print("Console: No pre-saved classifier found.")
        print("Console: Building a new classifier, this might take a while..")

        classifier = svm.SVC(kernel="rbf", C=5, gamma=0.05)
        classifier.fit(train_features, train_labels)

        with open("classifier.pkl", "wb") as writer:
            pickle.dump(classifier, writer)
            print("Console: Classifier saved to the memory.")

    print("Console: MNIST dataset performance: {:.4f}%".format(classifier.score(test_features, test_labels) * 100))
    return classifier


def read_image_and_its_data(img_dir, data_dir):
    name = os.path.basename(img_dir)
    img = cv2.imread(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')

    return name, img, data


def show_bounding_boxes(name, img):
    bounding_boxes = sr.detect_sudoku(img)
    cv2.rectangle(img, bounding_boxes[0][0], bounding_boxes[0][1], (0, 0, 255), 2)
    cv2.imshow(name, img)
    cv2.waitKey()


def obtain_sudoku_array(name, img, classifier, eigenvectors):
    sudoku_array = np.zeros((9, 9), dtype=np.uint8)

    try:
        sudoku_array = sr.recognize_sudoku(img, classifier, eigenvectors)
    except ValueError:  # The case where sudoku cannot be found in the image
        print("Console: No sudoku found in " + name + ".")

    return sudoku_array


def main():
    train_images, train_labels, test_images, test_labels = load_mnist_dataset()
    train_features, eigenvectors = apply_pca_to_mnist(train_images)

    test_images = (test_images - np.mean(test_images, axis=0))  # Subtract mean
    test_features = np.dot(test_images, eigenvectors)  # Dot product

    classifier = build_classifier(train_features, train_labels, test_features, test_labels)

    cumulative_accuracy = 0
    num_images = len(glob.glob(IMAGE_DIRS))

    # Loop over all images
    print("Console: Testing accuracy on Sudoku dataset, this might take a while..")
    for img_dir, data_dir in zip(glob.glob(IMAGE_DIRS), glob.glob(DATA_DIRS)):
        name, img, data = read_image_and_its_data(img_dir, data_dir)

        # Uncomment this section if you would like to see resulting bounding boxes.
        # show_bounding_boxes(name, img)

        sudoku_array = obtain_sudoku_array(name, img, classifier, eigenvectors)

        # Evaluate result for current image
        detection_accuracy_array = data == sudoku_array
        acc_percentage = np.sum(detection_accuracy_array) / detection_accuracy_array.size
        cumulative_accuracy = cumulative_accuracy + acc_percentage

        print("Console: " + name + " accuracy : " + str(acc_percentage * 100) + "%")

    # Average accuracy over all images in the dataset
    average_accuracy = cumulative_accuracy / num_images
    print("Console: Sudoku dataset performance: " + str(average_accuracy * 100) + "%")


if __name__ == '__main__':
    main()

