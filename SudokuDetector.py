""" Batuhan Erden S004345 Department of Computer Science """


import cv2
import numpy as np


class Kernel(object):

    def __init__(self, image, blur_boost=0, block_size=11):
        self.image = image
        self.blur_boost = blur_boost
        self.block_size = block_size
        self.blur_kernel_size, self.morphology_kernel = None, None

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

    def calculate_parameters(self):
        blurriness_level = self.get_blurriness_level(self.image) + self.blur_boost

        self.blur_kernel_size = (5 + blurriness_level, 5 + blurriness_level) \
            if not blurriness_level % 2 \
            else (4 + blurriness_level, 4 + blurriness_level)
        self.morphology_kernel = np.ones((3 + blurriness_level, 3 + blurriness_level), np.uint8)

    def get_image_filtered(self, image):
        self.calculate_parameters()

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
        rectangles, max_area, max_approx = [], 0., []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                epsilon = e * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # It's a rectangle
                    rectangles.append(contour)

                    if area >= max_area:
                        max_area = area
                        max_approx = approx

        return rectangles, max_approx

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

    def get_sudoku_rectangles(self, sudoku_frame, rectangles):
        sudoku_rectangles = []
        (X1, Y1), (X2, Y2) = self.get_points_of_rectangle(sudoku_frame)

        for rectangle in rectangles:
            (x1, y1), (x2, y2) = self.get_points_of_rectangle(rectangle)

            if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                sudoku_rectangles.append(rectangle)

        sudoku_rectangles = self.deepsort_rectangles(sudoku_rectangles)
        return sudoku_rectangles

    def get_sudoku_from_image(self):
        contours = self.find_contours_in_image()
        rectangles, max_approx = self.get_rectangles_from_contours(contours)

        sudoku_frame = self.get_sudoku_frame(rectangles)

        # Update block size
        if len(sudoku_frame) == 0:
            if self.kernel.block_size <= 101:
                self.kernel.block_size += 10
                return self.get_sudoku_from_image()
            else:
                return None, None, None, False

        sudoku_rectangles = self.get_sudoku_rectangles(sudoku_frame, rectangles)

        # Update blur boost
        if len(sudoku_rectangles) != 81:
            if self.kernel.blur_boost <= 10:
                self.kernel.blur_boost += 2

                _, next_sudoku_rectangles, _, _ = self.get_sudoku_from_image()

                if next_sudoku_rectangles is None:
                    next_sudoku_rectangles = []

                if len(sudoku_rectangles) <= len(next_sudoku_rectangles):
                    sudoku_rectangles = next_sudoku_rectangles
            else:
                return None, sudoku_rectangles, None, False

        return sudoku_frame, sudoku_rectangles, max_approx, True

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

