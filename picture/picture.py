import numpy as np
import cv2


def get_image_webcam():
    cam = cv2.VideoCapture(1)
    s, image = cam.read()
    cv2.waitKey(0)
    cam.release()
    return image


def transform_board(image, save=False):
    path = "./images/"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cimg = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 2, 100, param1=50,
                               param2=30, minRadius=40, maxRadius=55)
    if circles is None:
        return False

    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=150, maxLineGap=10)
    if lines is None:
        return False

    x1, y1, x2, y2 = lines[0][0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    circles = np.round(circles[0, :]).astype("int")
    # averageX = np.average(circles[:, 0])
    # averageY = np.average(circles[:, 1])
    # print(circles)
    # print(averageX)
    # print(averageY)

    for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            if save:
                cv2.imwrite("{}{}.png".format(path, x), image)

    return image


def display_picture(image):
        cv2.imshow("output transform", image)
        cv2.waitKey(0)


def transform_video():
    cap = cv2.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', transform_board(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
