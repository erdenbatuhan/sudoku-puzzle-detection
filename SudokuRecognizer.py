""" Batuhan Erden S004345 Department of Computer Science """


import cv2
import numpy as np
from SudokuDetector import SudokuDetector


def convert_labels(labels):
    labels_converted = []

    for label in labels:
        labels_converted.append([index for index, item in enumerate(label) if item][0])

    return labels_converted


def detect_sudoku(img):
    sudoku_detector = SudokuDetector(img)
    sudoku_frame, sudoku_rectangles, max_approx, found = sudoku_detector.get_sudoku_from_image()

    if not found:
        return [], []

    # Uncomment this section if you would like to see the sudoku with rectangles found.
    # sd.draw_sudoku()
    # cv2.imshow("Sudoku with rectangles found", img)
    # cv2.waitKey(0)

    bounding_boxes = []
    for each in [sudoku_frame] + sudoku_rectangles:
        x, y, w, h = cv2.boundingRect(each)
        x_left, y_left, x_right, y_right = x, y, x + w, y + h

        bounding_boxes.append([(x_left, y_left), (x_right, y_right)])

    # for b in bounding_boxes:
    #     (x_left, y_left), (x_right, y_right) = b
    #     cv2.rectangle(img, (x_left, y_left), (x_right, y_right), color=(0, 0, 255), thickness=2)

    return bounding_boxes, max_approx


def push_black_image(samples, non_zero_counts):
    black_img = np.zeros((28, 28))

    samples.append(black_img)
    non_zero_counts.append(0)


def get_samples_from_bounding_boxes(img, bounding_boxes, y_left, motion):
    samples = []
    non_zero_counts = []

    c, prev = 0, y_left
    for bounding_box in bounding_boxes:
        (x_left, y_left), (x_right, y_right) = bounding_box
        box = img[int(y_left): int(y_right), int(x_left):int(x_right)]

        # Crop the image again to obtain the digit only
        (h, w) = box.shape[:2]
        box = box[int(h * .125): int(h * .875), int(w * .125): int(w * .875)]

        # Preprocess the image
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        threshold = cv2.adaptiveThreshold(blur, 255, 0, 1, 11, 7)
        digit = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, (5, 5))

        # Count non-zero pixels in order to decide the content
        non_zero_counts.append(cv2.countNonZero(digit))

        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = digit / 255  # Division is needed because MNIST dataset is normalized between [0, 1]

        if prev is not None and y_left - prev >= (y_right - y_left) / 2:
            # Assign 0 for the rectangles that could not be found
            while c < 9:
                push_black_image(samples, non_zero_counts)
                c += 1

            c = 0

        samples.append(digit)
        c += 1

        prev = y_left

    # Assign 0 for the rectangles that could not be found
    while len(samples) < 81:
        push_black_image(samples, non_zero_counts)

    return samples, non_zero_counts


def get_samples_from_warped(warped, motion):
    params = {
        "blurSize": 1 if motion else 5,
        "blockSize": 51 if motion else 11
    }

    samples = []
    non_zero_counts = []

    h, w, _ = warped.shape
    dy, dx = h / 9, w / 9

    for i in range(0, 9):
        for j in range(0, 9):
            y = i * dy
            x = j * dx

            box = warped[int(y): int(y + dy), int(x): int(x + dx)]

            # Crop the image again to obtain the digit only
            (h, w) = box.shape[:2]
            box = box[int(h * .1): int(h * .9), int(w * .1): int(w * .9)]

            # Preprocess the image
            gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, params["blurSize"])
            threshold = cv2.adaptiveThreshold(blur, 255, 0, 1, params["blockSize"], 7)

            if not motion:
                digit = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, (5, 5))
            else:
                digit = cv2.erode(threshold, np.ones((3, 3), np.uint8), iterations=1)

            # Count non-zero pixels in order to decide the content
            non_zero_counts.append(cv2.countNonZero(digit))

            digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            digit = digit / 255  # Division is needed because MNIST dataset is normalized between [0, 1]

            samples.append(digit)

    return samples, non_zero_counts


def recognize_sudoku(img, warped, model, motion=False):
    sudoku_array = np.zeros((9, 9, 3), dtype=np.uint8)
    bounding_boxes, y_left = [], 0.

    # First bounding box belongs to the sudoku frame while other bounding boxes belong to the sudoku cells
    try:
        bounding_boxes, _ = detect_sudoku(img)
        (_, y_left), _ = bounding_boxes.pop(0)
    except IndexError:
        sudoku_array -= 1

    samples, non_zero_counts = get_samples_from_bounding_boxes(img, bounding_boxes, y_left, motion)
    non_zero_threshold = 50

    if np.sum(sudoku_array) != 0 or len(samples) > 81:
        sudoku_array = np.zeros((9, 9, 3), dtype=np.uint8)

        samples, non_zero_counts = get_samples_from_warped(warped, motion)
        non_zero_threshold = 100 if np.mean(non_zero_counts) < 500 else 500

    samples = np.reshape(samples, (-1, 28, 28, 1))

    for i in range(len(samples)):
        sample = np.array([samples[i]])
        predictions = []

        confidences = model.predict(sample, verbose=0)[0]  # Obtain confidences for each digit
        # confidences[0] = -np.inf  # Make 0's confidence -inf

        for _ in range(3):
            next_prediction = np.argmax(confidences)  # Get next predicted digit
            predictions.append(next_prediction)       # Append next predicted digit to predictions
            confidences[next_prediction] = -np.inf    # Make next predicted digit's confidence -inf

        sudoku_array[int(i / 9), i % 9] = predictions

    return sudoku_array


# # PCA on MNIST dataset, this function should return PCA transformation for mnist dataset.
# def apply_pca(X, e):
#     print("Console: Applying PCA on MNIST dataset..")
#     chosen_eigenvectors = None
#
#     # Subtract mean
#     X = (X - np.mean(X, axis=0))
#
#     eigenvalues, eigenvectors = np.linalg.eig(np.dot(X.T, X))  # Covariance Matrix
#     eigenvalues /= eigenvalues.sum()  # Normalization
#
#     eigen_sorted_ind = eigenvalues.argsort()
#     eigenvalues_sum = 0.
#
#     for i in reversed(range(len(eigen_sorted_ind))):
#         eigenvalues_sum += eigenvalues[eigen_sorted_ind[i]]
#         if eigenvalues_sum >= e:
#             chosen_eigenvectors = eigenvectors[:, eigen_sorted_ind[i + 1:]]
#             break
#
#     print("Console: PCA applied on MNIST dataset.")
#     return chosen_eigenvectors
#
#
# def get_image_rotated(img, angle):
#     if angle == 0:
#         return img
#
#     (h, w) = img.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     return cv2.warpAffine(img, M, (nW, nH))
#
#
# def directly_detec_sudoku(img):
#     # variables for the biggest detected contour expected to be Sudoku
#     maxContour, maxArea, maxPer = [], 0, 0
#
#     # smallFrame -> Grayscale -> Blur -> Threshold
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     threshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 51, 1)
#     cntOutIm, cnt, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     if cnt != []:
#         for a in cnt:
#             x, y, w, h = cv2.boundingRect(a)
#             if (cv2.arcLength(a, True) >= maxPer and w / h > 0.75 and w / h < 1.25 and cv2.contourArea(a) >= maxArea):
#                 maxPer = cv2.arcLength(a, True)
#                 maxContour = a
#                 maxArea = cv2.contourArea(a)
#
#     maxX, maxY, maxW, maxH = cv2.boundingRect(maxContour)
#     topLeftx, topLefty, botRightx, botRighty = maxX, maxY, maxX + maxW, maxY + maxH
#     boundingBox = [(topLeftx, topLefty), (botRightx, botRighty)]
#     approx = cv2.approxPolyDP(maxContour, 0.02 * maxPer, True)
#     maxContour = approx
#     return boundingBox, maxContour

