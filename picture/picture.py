import numpy as np
import cv2
import uuid
import os


def get_webcam_port():
    return 1


def get_image_webcam():
    cam = cv2.VideoCapture(get_webcam_port())
    image = None

    while(True):
        s, frame = cam.read()
        cv2.imshow('frame', crop_board(frame))
        if is_q_pressed():
            image = frame
            break

    cam.release()
    return image


def detect_and_crop_pit(image, number_pebble="None", save=False):
    board_image = crop_board(image)
    return crop_pit(board_image, number_pebble, save)


def crop_pit(image, number_pebble="None", save=False):
    path_save_image = os.path.join("images", number_pebble)
    if not os.path.exists(path_save_image):
        os.makedirs(path_save_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 100, param1=50,
                               param2=30, minRadius=40, maxRadius=55)
    if circles is None:
        return False

    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        if save:
            crop_image = crop_picture(image, x, y)
            cv2.imwrite(os.path.join(path_save_image,
                        "{}.png".format(uuid.uuid1())),
                        crop_image)
        else:
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

    return image


def crop_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Add mask black on white color
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([180, 180, 180])
    mask = cv2.inRange(gray, lower_blue, upper_blue)

    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

    # Get first large area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    # draw contours if you want debug
    # cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)

    # Draw rectangle and crop
    x, y, width, height = cv2.boundingRect(cnts[0])
    return image[y:y+height, x:x+width]


def crop_picture(image, x, y):
    cropSize = (100, 100)
    cropCoords = (max(0, y - (cropSize[0] // 2)),
                  min(image.shape[0], y + (cropSize[0] // 2)),
                  max(0, x - (cropSize[1] // 2)),
                  min(image.shape[1], x + (cropSize[1] // 2)))
    return image[cropCoords[0]:cropCoords[1], cropCoords[2]:cropCoords[3]]


def display_picture(image, title):
        cv2.imshow(title, image)


def detect_and_crop_pit_video():
    cap = cv2.VideoCapture(get_webcam_port())

    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', detect_and_crop_pit(frame))
        if is_q_pressed():
            break

    cap.release()
    cv2.destroyAllWindows()


def is_q_pressed():
    return cv2.waitKey(1) & 0xFF == ord('q')
