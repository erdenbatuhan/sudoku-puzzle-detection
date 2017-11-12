""" Batuhan Erden S004345 Department of Computer Science """


from glob import glob
import numpy as np
import math
import cv2


PATH_TO_IMAGES_FOLDER = "./images/"


class Kernel:

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


class SudokuDetector:

    def __init__(self, image):
        self.image = image

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

    @staticmethod
    def get_points_of_rectangle(rectangle):
        (x, y, w, h) = cv2.boundingRect(rectangle)
        return (x, y), (x + w, y + h)


class RectangleDetector(SudokuDetector):

    def __init__(self, image):
        SudokuDetector.__init__(self, image)

        self.kernel = Kernel(image)
        self.min_area = self.get_min_area()

    def get_min_area(self):
        (h, w) = self.image.shape[:2]
        return (w * h) / 500

    def draw_rectangles(self):
        rectangles = self.get_rectangles_from_image()

        if len(rectangles) == 0:
            return

        for rectangle in rectangles[1:len(rectangles)]:
            self.draw_rectangle(rectangle, (255, 0, 0))

        self.draw_rectangle(rectangles[0], (0, 0, 255))

    def get_rectangles_from_image(self):
        contours = self.find_contours_in_image()
        rectangles = self.get_rectangles_from_contours(contours)

        sudoku_frame = self.get_sudoku_frame(rectangles)

        if len(sudoku_frame) == 0:
            if self.kernel.block_size <= 101:
                self.kernel.block_size += 10
                return self.get_rectangles_from_image()
            else:
                return []

        return [sudoku_frame] + rectangles

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

    def find_contours_in_image(self, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        image_filtered = self.kernel.get_image_filtered(self.image)
        _, contours, _ = cv2.findContours(image_filtered, mode, method)  # Find the contours in the image

        return contours

    def draw_rectangle(self, rectangle, color, thickness=3):
        (x, y, w, h) = cv2.boundingRect(rectangle)
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color, thickness)


def read_images_from():
    images = []
    print("Progress Report: Reading images from %s" % PATH_TO_IMAGES_FOLDER)

    for file_name in glob(PATH_TO_IMAGES_FOLDER + "/*"):
        if "data" not in file_name and ".py" not in file_name:
            images.append((file_name, cv2.imread(file_name)))

    if len(images) == 0:
        print("No images found! Good Bye..")
        exit(0)

    print("Progress Report: %d images are read!" % len(images))
    return images


def get_images_processed(images):
    images_processed = []
    print("Progress Report: Processing those images..")

    for name, image in images:
        rectangle_detector = RectangleDetector(image)
        rectangle_detector.draw_rectangles()

        images_processed.append((name, rectangle_detector.image))

    print("Progress Report: Images are processed!")
    return images_processed


def show_images_processed(images_processed):
    print("Progress Report: Showing images..")

    for name, image_processed in images_processed:
        cv2.imshow(name, get_image_resized(image_processed))

        if cv2.waitKey(0) & 0xff == 27:
            print("Progress Report: You've pressed ESC Key, Good Bye!")

            cv2.destroyAllWindows()
            exit(0)

        cv2.destroyAllWindows()

    print("Progress Report: All images are shown, Good Bye!")


def get_image_resized(image, max_size=600, inverse_scale=0.05):
    while True:
        (h, w) = image.shape[:2]

        if h <= max_size and w <= max_size:
            break

        scale = 1 - inverse_scale
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return image


def main():
    images = read_images_from()
    images_processed = get_images_processed(images)

    show_images_processed(images_processed)


if __name__ == "__main__":
    main()

