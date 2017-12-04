#DO NOT IMPORT ANY OTHER LIBRARY.
import cv2
import numpy as np

# Define your own functions, classes, etc. here:
#
#


# PCA on MNIST dataset, this function should return PCA transformation for mnist dataset.
# You can freely modify the signature of the function.
def mnistPCA(train_images, e=0.9):
    train_images = (train_images - np.mean(train_images, axis=0))

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(train_images.T, train_images))
    eigen_sorted_ind = eigenvalues.argsort()[::-1]

    eigenvalues /= eigenvalues.sum()

    eigenvalues_sum = 0
    eigenvectors_selected = eigenvectors

    for i in range(len(eigen_sorted_ind)):
        eigenvalues_sum += eigenvalues[eigen_sorted_ind[i]]
        if eigenvalues_sum >= e:
            eigenvectors_selected = eigenvectors[:, eigen_sorted_ind[0:i]]
            break

    return eigenvectors_selected


def RecognizeSudoku(img):
    #  An example of SudokuArray is given below :
    #   0 0 0 7 0 0 0 8 0
    #   0 9 0 0 0 3 1 0 0
    #   0 0 6 8 0 5 0 7 0
    #   0 2 0 6 0 0 0 4 9
    #   0 0 0 2 0 0 0 5 0
    #   0 0 8 0 4 0 0 0 7
    #   0 0 0 9 0 0 0 3 0
    #   3 7 0 0 0 0 0 0 6
    #   1 0 5 0 0 4 0 0 0
    # Where 0 represents empty cells in sudoku puzzle. SudokuArray must be a numpy array.

    # WRITE YOUR SUDOKU RECOGNIZER CODE HERE.
    # code code code
    # code code code
    # code code code
    # maybe more code

    SudokuArray = []
    return SudokuArray



# Returns Sudoku puzzle bounding box following format [(topLeftx, topLefty), (bottomRightx, bottomRighty)]
# (You can change the body of this function anyway you like.)
def detectSudoku(img):
    # variables for the biggest detected contour expected to be Sudoku
    maxContour, maxArea, maxPer = [], 0, 0

    # smallFrame -> Grayscale -> Blur -> Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    threshold = cv2.adaptiveThreshold(blur, 255, 0, 1, 11, 7)

    # find all contours in the thresholded image
    cntOutIm, cnt, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the largest contour by comparing their size in terms of area and perimeter
    # it also checks the width height ratio of the biggest contour
    if cnt != []:
        for a in cnt:
            x, y, w, h = cv2.boundingRect(a)
            if (cv2.arcLength(a, True) >= maxPer and w / h > 0.75 and w / h < 1.25 and cv2.contourArea(a) >= maxArea):
                maxPer = cv2.arcLength(a, True)
                maxContour = a
                maxArea = cv2.contourArea(a)

    maxX, maxY, maxW, maxH = cv2.boundingRect(maxContour)

    topLeftx, topRighty, botLeftx, botLefty = maxX, maxY, maxX + maxW, maxY + maxH
    boundingBox = [(topLeftx, topRighty), (botLeftx, botLefty)]

    return boundingBox






