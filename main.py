#DO NOT IMPORT ANY OTHER LIBRARY.
import numpy as np
import glob
import cv2
import os
from matplotlib import pyplot as plt

import SudokuRecognizer as sr
from mnist_web import mnist


# Define your functions here if required :
def get_nearest_sample_index(sample, train_data):
    # OpenCV's Support Vector Machine
    dist_matrix = sample - train_data
    dist_square = dist_matrix ** 2
    dist_sums = dist_square.sum(axis=1)
    distance_vector = np.sqrt(dist_sums)

    return distance_vector.argmin()


def get_label(ind):
    l = 0
    for i in ind:
        l += 1
        if i:
            return l


# Set dataset path before start example  '/Home/sudoku_dataset-master' :
sudoku_dataset_dir = '/Users/barisozcan/Documents/CS423 Computer Vision/sudoku_dataset-master'
MNIST_dataset_dir = "/Users/barisozcan/Documents/MNIST_dataset"

# Load MNIST Dataset:
MNIST_dataset_dir = "./MNIST_dataset"
train_images, train_labels, test_images, test_labels = mnist(path="./MNIST_dataset")

# Apply PCA to MNIST

print(len(test_images))
eigenvectors = sr.mnistPCA(train_images)

train_data = np.dot(train_images, eigenvectors)
#train_images = (train_images - np.mean(train_images, axis=0))
sample_data = np.dot(test_images, eigenvectors)

found = 0
for i in range(len(test_images)):
    sample = sample_data[i]
    delta = get_nearest_sample_index(sample, train_data)

    train_label = get_label(train_labels[delta])
    sample_label = get_label(test_labels[i])

    if train_label == sample_label:
        found += 1


accuracy = found / len(test_images)
print("Accuracy: {}%".format(accuracy))



# use sr.mnistPCA() that you applier for transformation
# classify test set with any method you choose (hint simplest one : nearest neighbour)
# report the outcome
# Calculate confusion matrix, false postivies/negatives.
# print(reporting_results)



image_dirs = sudoku_dataset_dir + '/images/image*.jpg'
data_dirs = sudoku_dataset_dir + '/images/image*.dat'
IMAGE_DIRS = glob.glob(image_dirs)
DATA_DIRS = glob.glob(data_dirs)
len(IMAGE_DIRS)


#Define your variables etc. outside for loop here:

# Accumulate accuracy for average accuracy calculation.
cumulativeAcc = 0

## Loop over all images
for img_dir, data_dir in zip(IMAGE_DIRS, DATA_DIRS):

    #Define your variables etc.:
    image_name = os.path.basename(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
    img = cv2.imread(img_dir)


    # detect sudoku puzzle:
    boundingBox = sr.detectSudoku(img)

    # Uncomment this section if you would like to see resulting bounding boxes.
    # cv2.rectangle(img, boundingBox[0], boundingBox[1], (0, 0, 255), 2)
    # cv2.imshow(image_name, img)
    # cv2.waitKey()

    # Recognize digits in sudoku puzzle :
    sudokuArray = sr.RecognizeSudoku(img)

    # Evaluate Result for current image :

    detectionAccuracyArray = data == sudokuArray
    accPercentage = np.sum(detectionAccuracyArray)/detectionAccuracyArray.size
    cumulativeAcc = cumulativeAcc + accPercentage
    print(image_name + " accuracy : " + accPercentage.__str__() + "%")

# Calculate confusion matrix, false postivies/negatives for Sudoku dataset here.



# Average accuracy over all images in the dataset :
averageAcc  = cumulativeAcc/len(IMAGE_DIRS)
print("dataset performance : " + averageAcc.__str__() + "%")

