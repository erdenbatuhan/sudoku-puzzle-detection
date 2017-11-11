from glob import glob
import numpy as np
import cv2


PATH_TO_IMAGES_FOLDER = "./images/"


class Kernel:

    def __init__(self):
        pass

    def get_image_filtered(self, image):
        morphology_kernel = np.ones((3, 3), np.uint8)

        if self.is_image_blurry(image):
            morphology_kernel = np.ones((6, 6), np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Make the image gray
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur on the image
        thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)  # Threshold the image
        morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morphology_kernel)  # Apply morphology operations

        return morphology

    @staticmethod
    def is_image_blurry(image, threshold=120):
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        return True if variance <= threshold else False


class RectangleDetector:

    def __init__(self):
        self.kernel = Kernel()

    def get_rectangles_from_image(self, image, min_area=1000, epsilon_multiplier=0.04):
        outer_rectangle, rectangles = None, []
        index, max_index, max_area = 0, 0, 0

        for contour in self.get_contours_from_image(image):
            area = cv2.contourArea(contour)

            if area >= min_area:
                arc_length = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_multiplier * arc_length, True)

                if len(approx) == 4:  # It's a rectangle
                    if area > max_area:
                        outer_rectangle = contour
                        max_index = index
                        max_area = area

                    rectangles.append(contour)
                    index += 1

        if len(rectangles) is not 0:
            rectangles.pop(max_index)  # Remove outer rectangle from rectangles

        return outer_rectangle, rectangles

    def get_contours_from_image(self, image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        image_filtered = self.kernel.get_image_filtered(image)
        _, contours, _ = cv2.findContours(image_filtered, mode, method)  # Find the contours in the image

        return contours


class SudokuDetector:

    def __init__(self, image):
        self.rectangle_detector = RectangleDetector()
        self.image = image

    def draw_sudoku(self, thickness=3):
        sudoku = self.get_sudoku()
        color = (0, 0, 255)  # Color is red for first rectangle in the list (outer rectangle)

        for box in sudoku:
            (x, y, w, h) = cv2.boundingRect(box)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, thickness)

            color = (255, 0, 0)  # Color is blue for the rest (inner rectangles)

    def get_sudoku(self):
        outer_rectangle, inner_rectangles = self.rectangle_detector.get_rectangles_from_image(self.image)
        sudoku = [outer_rectangle]

        (X1, Y1), (X2, Y2) = self.get_points_of_rectangle(outer_rectangle)

        for inner_rectangle in inner_rectangles:
            (x1, y1), (x2, y2) = self.get_points_of_rectangle(inner_rectangle)

            if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                sudoku.append(inner_rectangle)

        if len(sudoku) <= 50:  # The image doesn't contain sudoku
            return []

        return sudoku

    @staticmethod
    def get_points_of_rectangle(rectangle):
        (x, y, w, h) = cv2.boundingRect(rectangle)
        return (x, y), (x + w, y + h)


def read_images_from():
    images = []
    print("Progress Report: Reading images from %s" % PATH_TO_IMAGES_FOLDER)

    for file_name in glob(PATH_TO_IMAGES_FOLDER + "/*"):
        if "data" not in file_name and ".py" not in file_name:
            images.append(cv2.imread(file_name))

    if len(images) == 0:
        print("No images found! Good Bye..")
        exit(0)

    print("Progress Report: %d images are read!" % len(images))
    return images


def get_images_processed(images):
    images_processed = []
    print("Progress Report: Processing those images..")

    for image in images:
        sudoku_detector = SudokuDetector(image)
        sudoku_detector.draw_sudoku()

        images_processed.append(sudoku_detector.image)

    print("Progress Report: Images are processed!")
    return images_processed


def show_images_processed(images_processed):
    print("Progress Report: Showing images..")

    for image_processed in images_processed:
        cv2.imshow("Sudoku", get_image_resized(image_processed))

        if cv2.waitKey(0) & 0xff == 27:
            print("Progress Report: You've pressed ESC Key, Good Bye!")

            cv2.destroyAllWindows()
            exit(0)

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

