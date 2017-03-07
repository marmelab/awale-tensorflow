import numpy as np
import cv2
import uuid
import os


def get_image_webcam():
    cam = cv2.VideoCapture(1)
    s, image = cam.read()
    cv2.waitKey(0)
    cam.release()
    return image


def transform_board(image, numberPebble="None", save=False):
    path = "./images/{}".format(numberPebble)
    if not os.path.exists(path):
        os.makedirs(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cimg = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 2, 100, param1=50,
                               param2=30, minRadius=40, maxRadius=55)
    if circles is None:
        return False

    # edges = cv2.Canny(gray, 50, 200)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
    #                         minLineLength=150, maxLineGap=10)
    # if lines is None:
    #     return False
    #
    # x1, y1, x2, y2 = lines[0][0]
    # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        if save:
            crop_image = crop_picture(image, x, y)
            cv2.imwrite(os.path.join(path, "{}.png".format(uuid.uuid1())),
                        crop_image)
        else:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    return image


def crop_picture(image, x, y):
    cropSize = (100, 100)
    cropCoords = (max(0, y-cropSize[0]//2),
                  min(image.shape[0], y+cropSize[0]//2),
                  max(0, x-cropSize[1]//2),
                  min(image.shape[1], x+cropSize[1]//2))
    return image[cropCoords[0]:cropCoords[1], cropCoords[2]:cropCoords[3]]


def display_picture(image):
        cv2.imshow("output transform", image)
        cv2.waitKey(0)


def transform_video():
    cap = cv2.VideoCapture(1)

    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', transform_board(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
