""" Batuhan Erden S004345 Department of Computer Science """


import numpy as np
import time
import glob
import cv2
import os
import SudokuRecognizer
from SudokuSolver import SudokuSolver
from mnist_web import mnist

import keras
from keras.preprocessing import image
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# Set dataset paths before start. For example: '/Home/sudoku_dataset-master'
SUDOKU_DATASET_DIR = "/Users/batuhan/Documents/workspace/Sudoku-Puzzle-Detection/dataset"
MNIST_DATASET_DIR = "/Users/batuhan/Documents/workspace/Sudoku-Puzzle-Detection/MNIST_dataset"

VIDEO_DIRS = SUDOKU_DATASET_DIR + "/test_videos/test-*.mp4"
IMAGE_DIRS = SUDOKU_DATASET_DIR + "/images/image*.jpg"
DATA_DIRS = SUDOKU_DATASET_DIR + "/images/image*.dat"

IS_VIDEO = False  # Make IS_VIDEO False if you want to loop over all images instead of looping over all videos


def augment_mnist(images, labels):
    t0 = time.time()
    print("Console: Augmenting the MNIST dataset, this will take approx. 20 seconds..")

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        if label == 0:
            # Black Image
            augmented_images.append(np.zeros(image.shape, dtype=np.float))
            augmented_labels.append(label)
        else:
            image_2D = np.reshape(image, (28, 28, 1))

            # Actual Image
            augmented_images.append(image)
            augmented_labels.append(label)

            # Random Rotation
            augmented_images.append(keras.preprocessing.image.random_rotation(image_2D, 20, row_axis=0, col_axis=1,
                                                                              channel_axis=2).reshape((784, )))
            augmented_labels.append(label)

            # Random Shift
            augmented_images.append(keras.preprocessing.image.random_shift(image_2D, 0.2, 0.2, row_axis=0, col_axis=1,
                                                                           channel_axis=2).reshape((784, )))
            augmented_labels.append(label)

    print("Console: MNIST dataset is augmented ({} images and {} labels) and took {:.2f} seconds.".
          format(len(augmented_images), len(augmented_labels), time.time() - t0))
    return np.array(augmented_images), np.array(augmented_labels)


def build_model(X, y):
    t0 = time.time()
    print("Console: Building the model, this might take a while..")

    X = np.reshape(X, (-1, 28, 28, 1))
    y = keras.utils.to_categorical(y, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=12, verbose=1)

    print("Console: The model is built.")
    print("Console: Took {:.2f} minutes.".format((time.time() - t0) / 60))

    save_model(model=model, filepath="model.h5", overwrite=True)
    print("Console: The model is saved to the memory.")

    return model


def warp(img, bounding_boxes=None, max_approx=None):
    if bounding_boxes is None and max_approx is None:
        bounding_boxes, max_approx = SudokuRecognizer.detect_sudoku(img)

        if len(bounding_boxes) + len(max_approx) == 0:
            return img, [], [], [], False

    try:
        def get_distance_between(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        corners = max_approx.reshape(4, 2)

        vertical_sum = corners.sum(axis=1)
        vertical_diff = np.diff(corners, axis=1)

        rect = np.array([corners[np.argmin(vertical_sum)], corners[np.argmin(vertical_diff)],
                         corners[np.argmax(vertical_sum)], corners[np.argmax(vertical_diff)]], dtype=np.float32)

        # p1: Top Left, p2: Top Right, p3: Bottom Right and p4: Bottom Left
        (p1, p2, p3, p4) = rect  # (Top Left, Top Right, Bottom Right, Bottom Left)

        top_width = get_distance_between(p1, p2)
        bottom_width = get_distance_between(p3, p4)
        left_height = get_distance_between(p1, p4)
        right_height = get_distance_between(p2, p3)

        width = max(int(top_width), int(bottom_width))
        height = max(int(left_height), int(right_height))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(img, M, (width, height))
        return warped, max_approx, rect, dst, True
    except ValueError:
        (x_left, y_left), (x_right, y_right) = bounding_boxes[0]
        warped = img[y_left: y_right, x_left: x_right]
        rect, dst = [], []

        return warped, max_approx, rect, dst, False


def get_digit_box(digit, shape, color):
    digit_box = np.zeros((shape[0], shape[1], 3), dtype=np.float) + 255
    frame_size = int(digit_box.shape[0] / 28)

    digit_box[frame_size:digit_box.shape[0] - frame_size, frame_size:digit_box.shape[1] - frame_size] = 0
    cv2.putText(digit_box, str(digit), (int(digit_box.shape[0] / 2.6), int(digit_box.shape[1] / 1.6)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return digit_box


def draw_sudoku(img, sudoku_array, original_digits):
    h, w, _ = img.shape
    dy, dx = h / 9, w / 9

    for i in range(0, 9):
        for j in range(0, 9):
            y = i * dy
            x = j * dx

            box_shape = img[int(y): int(y + dy), int(x): int(x + dx)].shape
            color = (255, 255, 255) if original_digits[i, j] else (0, 255, 0)

            digit_box = get_digit_box(sudoku_array[i, j], box_shape, color)
            img[int(y): int(y + dy), int(x): int(x + dx)] = digit_box

    return img


def inv_warp(img, warped, max_approx, rect, dst):
    height, width, _ = img.shape
    img = cv2.fillConvexPoly(img, max_approx, (0, 0, 0))

    M_inv = cv2.getPerspectiveTransform(dst, rect)
    inv_warped = cv2.warpPerspective(warped, M_inv, (width, height))

    return img + inv_warped


def main():
    try:
        model = load_model(filepath="model.h5", compile=True)
        print("Console: A pre-saved model is loaded.")
    except OSError:
        print("Console: Loading the MNIST dataset..")
        train_images, train_labels, test_images, test_labels = mnist(path=MNIST_DATASET_DIR)

        # Converting labels to integers
        train_labels = SudokuRecognizer.convert_labels(train_labels)
        test_labels = SudokuRecognizer.convert_labels(test_labels)

        # Concatenating the arrays
        images = np.concatenate((train_images, test_images))
        labels = np.concatenate((train_labels, test_labels))
        print("Console: MNIST dataset is loaded..")

        X, y = augment_mnist(images, labels)
        model = build_model(X, y)

    # SOLVE SUDOKUs
    if IS_VIDEO:  # Loop over all videos
        for video_dir in glob.glob(VIDEO_DIRS):
            video = cv2.VideoCapture(video_dir)
            warped = None

            while video.isOpened():
                ret, frame = video.read()

                if ret:
                    try:
                        warped, max_approx, rect, dst, is_warped = warp(frame)

                        if is_warped:
                            try:
                                sudoku_array = SudokuRecognizer.recognize_sudoku(frame, warped, model)
                            except ValueError:  # The case where sudoku cannot be found in the image
                                sudoku_array = np.zeros((9, 9, 3), dtype=np.uint8)
                                print("Console: No sudoku found.")

                            sudoku_array = sudoku_array[:, :, 0]  # Get only the very first predictions
                            original_digits = sudoku_array != 0

                            print(sudoku_array)
                            draw_sudoku(warped, sudoku_array, original_digits)

                            frame = inv_warp(frame, warped, max_approx, rect, dst)
                    except Exception:
                        pass

                    cv2.imshow("frame", frame)
                if (cv2.waitKey(30) & 0xFF == ord("q")) or not ret:
                    break

            video.release()
            cv2.destroyAllWindows()
    else:  # Loop over all images
        cumulative_accuracy = 0
        num_images = len(glob.glob(IMAGE_DIRS))

        print("Console: Testing accuracy on Sudoku dataset, this might take a while..")
        for img_dir, data_dir in zip(glob.glob(IMAGE_DIRS), glob.glob(DATA_DIRS)):
            try:
                name, img, data = os.path.basename(img_dir), cv2.imread(img_dir), \
                                  np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')

                # Uncomment this section if you would like to see resulting bounding boxes.
                # bounding_boxes = SudokuRecognizer.detect_sudoku(img)
                # cv2.rectangle(img, bounding_boxes[0][0], bounding_boxes[0][1], (0, 0, 255), 2)
                # cv2.imshow(name, img)
                # cv2.waitKey()

                warped, max_approx, rect, dst, is_warped = warp(img)
                cv2.imshow("warped_sudoku", warped)

                if is_warped:
                    try:
                        sudoku_array = SudokuRecognizer.recognize_sudoku(img, warped, model)
                    except ValueError:  # The case where sudoku cannot be found in the image
                        sudoku_array = np.zeros((9, 9, 3), dtype=np.uint8)
                        print("Console: No sudoku found in " + name + ".")

                    sudoku_array = sudoku_array[:, :, 0]  # Get only the very first predictions
                    original_digits = sudoku_array != 0

                    print(sudoku_array)
                    draw_sudoku(warped, sudoku_array, original_digits)

                    img = inv_warp(img, warped, max_approx, rect, dst)
                    cv2.imshow("inv_warped_sudoku", img)

                    # Evaluate result for current image
                    detection_accuracy_array = data == sudoku_array
                    acc_percentage = np.sum(detection_accuracy_array) / detection_accuracy_array.size
                    cumulative_accuracy = cumulative_accuracy + acc_percentage

                    print("Console: " + name + " accuracy : " + str(acc_percentage * 100) + "%")
                    cv2.waitKey(0)
            except Exception:
                pass

        # Average accuracy over all images in the dataset
        average_accuracy = cumulative_accuracy / num_images
        print("Console: Sudoku dataset performance: " + str(average_accuracy * 100) + "%")


if __name__ == '__main__':
    main()


# def enrich_mnist_dataset(images, labels):
#     print("Console: Enriching the MNIST dataset, this might take a while..")
#
#     enriched_images = []
#     enriched_labels = []
#
#     for image, label in zip(images, labels):
#         enriched_images.append(image)
#         enriched_labels.append(label)
#
#         image = np.reshape(image, (-1, 28))
#
#         for angle in range(0, 360, 90):
#             if angle == 0:
#                 blur_image = cv2.medianBlur(image, 5)
#                 enriched_images.append(np.reshape(blur_image, 784))
#             else:
#                 enriched_images.append(np.reshape(ndimage.rotate(image, angle, reshape=False), 784))
#
#             enriched_labels.append(label)
#
#     enriched_images = np.array(enriched_images)
#     enriched_labels = np.array(enriched_labels)
#
#     print("Console: MNIST enriched.")
#     return enriched_images, enriched_labels
#
#
# def remove_zeros_from_mnist(images, labels):
#     if len(images) != len(labels):
#         print("Console: There was a problem loading the MNIST dataset.")
#         exit(0)
#
#     image_shape = images.shape[1:]
#     items_to_remove = []
#
#     for i in range(len(images)):
#         if labels[i] == 0:
#             images[i] = np.zeros(image_shape, dtype=np.float)
#             # items_to_remove.append(i)
#
#     # images = np.delete(images, items_to_remove, axis=0)
#     # labels = np.delete(labels, items_to_remove)
#
#     return images, labels

