""" Batuhan Erden S004345 Department of Computer Science """


import math
import cv2
import numpy as np


class Kernel(object):

    def __init__(self, image, block_size=11):
        self.image = image
        self.block_size = block_size
        self.blur_kernel_size, self.morphology_kernel = None, None

        self.calculate_parameters()

    def calculate_parameters(self):
        blurriness_level = self.get_blurriness_level(self.image)

        self.blur_kernel_size = (5 + blurriness_level, 5 + blurriness_level) \
            if not blurriness_level % 2 \
            else (4 + blurriness_level, 4 + blurriness_level)
        self.morphology_kernel = np.ones((3 + blurriness_level, 3 + blurriness_level), np.uint8)

    @staticmethod
    def get_blurriness_level(image):
        blurriness_level = 0
        variance = cv2.Laplacian(image, cv2.CV_64F).var()

        for threshold in [120, 100, 60, 15, 10]:
            if variance <= threshold:
                blurriness_level += 1
            else:
                break

        return blurriness_level

    def get_image_filtered(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Make the image gray
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)  # Apply Gaussian Blur on the image
        thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, self.block_size, 2)  # Threshold the image
        morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morphology_kernel)  # Apply morphology operations

        return morphology


class SudokuDetector(object):

    def __init__(self, image):
        self.image = image

        self.kernel = Kernel(image)
        self.min_area = self.get_min_area()

    @staticmethod
    def deepsort_rectangles(rectangles_in_sudoku):
        def sort_y_x(rectangle):
            x, y, w, h = cv2.boundingRect(rectangle)
            return y, x

        def sort_x(rectangle):
            x, y, w, h = cv2.boundingRect(rectangle)
            return x

        sorted_rectangles = []

        rectangles_in_sudoku.sort(key=sort_y_x)
        del rectangles_in_sudoku[0]  # Remove the biggest rectangle

        rows = [[]]
        prev, c = None, 0
        for r in rectangles_in_sudoku:
            x, y, w, h = cv2.boundingRect(r)

            if prev is not None and y - prev > h * .5:
                rows.append([])
                c += 1

            rows[c].append(r)
            prev = y

        for row in rows:
            row.sort(key=sort_x)

            for r in row:
                sorted_rectangles.append(r)

        return sorted_rectangles

    @staticmethod
    def get_points_of_rectangle(rectangle):
        (x, y, w, h) = cv2.boundingRect(rectangle)
        return (x, y), (x + w, y + h)

    def get_min_area(self):
        (h, w) = self.image.shape[:2]
        return (w * h) / 500

    def find_contours_in_image(self, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        image_filtered = self.kernel.get_image_filtered(self.image)
        _, contours, _ = cv2.findContours(image_filtered, mode, method)  # Find the contours in the image

        return contours

    def get_rectangles_from_contours(self, contours, e=0.04):
        rectangles = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                epsilon = e * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # It's a rectangle
                    rectangles.append(contour)

        return rectangles

    def get_sudoku_frame(self, rectangles, min_rectangle_count=50):
        possible_sudoku_frames = []

        for i in range(len(rectangles)):
            (X1, Y1), (X2, Y2) = self.get_points_of_rectangle(rectangles[i])
            rectangle_count = 0

            for j in range(len(rectangles)):
                (x1, y1), (x2, y2) = self.get_points_of_rectangle(rectangles[j])

                if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                    rectangle_count += 1

            if rectangle_count >= min_rectangle_count:
                possible_sudoku_frames.append((rectangles[i], rectangle_count))

        if len(possible_sudoku_frames) == 0:
            return []

        return sorted(possible_sudoku_frames, key=lambda x: x[1])[0][0]

    def get_sudoku_from_image(self):
        contours = self.find_contours_in_image()
        rectangles = self.get_rectangles_from_contours(contours)

        sudoku_frame = self.get_sudoku_frame(rectangles)

        if len(sudoku_frame) == 0:
            if self.kernel.block_size <= 101:
                self.kernel.block_size += 10
                return self.get_sudoku_from_image()
            else:
                return []

        rectangles_in_sudoku = []
        (X1, Y1), (X2, Y2) = self.get_points_of_rectangle(sudoku_frame)

        for rectangle in rectangles:
            (x1, y1), (x2, y2) = self.get_points_of_rectangle(rectangle)

            if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                rectangles_in_sudoku.append(rectangle)

        rectangles_in_sudoku = self.deepsort_rectangles(rectangles_in_sudoku)
        return sudoku_frame, rectangles_in_sudoku

    def draw_rectangle(self, rectangle, color, thickness=3):
        (x, y, w, h) = cv2.boundingRect(rectangle)
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color, thickness)

    def draw_sudoku(self):
        sudoku_frame, sudoku = self.get_sudoku_from_image()

        if len(sudoku) == 0:
            return

        for rectangle in sudoku:
            self.draw_rectangle(rectangle, (255, 0, 0))

        self.draw_rectangle(sudoku_frame, (0, 0, 255))


def get_nearest_euclidean_distance(sample, train_data):
    distances_squared = (sample - train_data) ** 2
    distances_sum = distances_squared.sum(axis=1)

    nearest = distances_sum.argmin()
    confidence = distances_sum[nearest]

    return nearest, confidence


def convert_labels(labels):
    labels_converted = []

    for label in labels:
        labels_converted.append([index for index, item in enumerate(label) if item][0])

    return labels_converted


# PCA on MNIST dataset, this function should return PCA transformation for mnist dataset.
def apply_pca(train_images, e=0.8):
    chosen_eigenvectors = None

    # Subtract mean
    train_data = (train_images - np.mean(train_images, axis=0))

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(train_data.T, train_data))
    eigenvalues /= eigenvalues.sum()  # Normalization

    eigen_sorted_ind = eigenvalues.argsort()
    eigenvalues_sum = 0.

    for i in reversed(range(len(eigen_sorted_ind))):
        eigenvalues_sum += eigenvalues[eigen_sorted_ind[i]]
        if eigenvalues_sum >= e:
            chosen_eigenvectors = eigenvectors[:, eigen_sorted_ind[i + 1:]]
            break

    return chosen_eigenvectors


def push_black_image(samples, non_zero_counts):
    black_img = np.zeros((28, 28))

    samples.append(black_img)
    non_zero_counts.append(0)


def get_samples_from_bounding_boxes(img, bounding_boxes, y_left):
    samples = []
    non_zero_counts = []

    c, prev = 0, y_left
    for bounding_box in bounding_boxes:
        (x_left, y_left), (x_right, y_right) = bounding_box
        box = img[int(y_left): int(y_right), int(x_left):int(x_right)]

        # Crop the image again to obtain the digit only
        (h, w) = box.shape[:2]
        box = box[int(h * .125): int(h * .875), int(w * .125): int(w * .875)]

        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        threshold = cv2.adaptiveThreshold(blur, 255, 0, 1, 11, 7)
        box = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, (5, 5))

        # Count non-zero pixels in order to decide the content
        non_zero_counts.append(cv2.countNonZero(box))

        box = cv2.resize(box, (28, 28), interpolation=cv2.INTER_AREA)
        box = box / 255  # Division is needed because MNIST dataset is normalized between [0, 1]

        if prev is not None and y_left - prev >= (y_right - y_left) / 2:
            # Assign 0 for the rectangles that could not be found
            while c < 9:
                push_black_image(samples, non_zero_counts)
                c += 1

            c = 0

        samples.append(box)
        c += 1

        prev = y_left

    # Assign 0 for the rectangles that could not be found
    while len(samples) < 81:
        push_black_image(samples, non_zero_counts)

    return samples, non_zero_counts


def get_image_rotated(img, angle):
    if angle == 0:
        return img

    rows, cols, _ = img.shape

    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 0.7)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))


def recognize_sudoku(img, classifier, eigenvectors):
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

    sudoku_array = np.zeros((9, 9), dtype=np.uint8) - 1
    bounding_boxes = detect_sudoku(img)

    # First bounding box belongs to the sudoku frame while other bounding boxes belong to the sudoku cells
    (x_left, y_left), (x_right, y_right) = bounding_boxes.pop(0)

    # Uncomment this section if you would like to see resulting bounding boxes.
    # sudoku = img[int(y_left): int(y_right), int(x_left):int(x_right)]
    # cv2.imshow("Sudoku", sudoku)
    # cv2.waitKey(0)

    samples, non_zero_counts = get_samples_from_bounding_boxes(img, bounding_boxes, y_left)

    samples = (samples - np.mean(samples, axis=0))  # Subtract mean
    sample_features = np.dot(np.reshape(samples, (81, 784)), eigenvectors)  # Dot product

    '''
    for i in range(len(samples)):
        if non_zero_counts[i] <= 50:
            SudokuArray[int(i / 9), i % 9] = 0
        else:
            nearest_index, value = get_nearest_euclidean_distance(samples[i], train_data)

            train_label = get_label(train_labels[nearest_index])
            SudokuArray[int(i / 9), i % 9] = train_label
    '''

    # ----------------------------------- SVM -----------------------------------
    for i in range(len(sample_features)):
        if non_zero_counts[i] <= 50:
            sudoku_array[int(i / 9), i % 9] = 0
        else:
            sample = sample_features[i].reshape((1, -1))
            sudoku_array[int(i / 9), i % 9] = classifier.predict(sample)
    # ---------------------------------------------------------------------------

    return sudoku_array


# Returns Sudoku puzzle bounding box following format: [(x_left, y_left), (x_right, y_right)]
def detect_sudoku(img):
    sd = SudokuDetector(img)
    sudoku_frame, sudoku = sd.get_sudoku_from_image()

    # Uncomment this section if you would like to see the sudoku with rectangles found.
    # sd.draw_sudoku()
    # cv2.imshow("Sudoku with rectangles found", img)
    # cv2.waitKey(0)

    bounding_boxes = []
    for each in [sudoku_frame] + sudoku:
        x, y, w, h = cv2.boundingRect(each)
        x_left, y_left, x_right, y_right = x, y, x + w, y + h

        bounding_boxes.append([(x_left, y_left), (x_right, y_right)])

    return bounding_boxes
