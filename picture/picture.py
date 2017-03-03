import numpy as np
import cv2


def get_image_webcam():
    cam = cv2.VideoCapture(1)
    s, image = cam.read()
    # file = "./images/webcam/test_image.png"
    # cv2.imwrite(file, img)
    cv2.waitKey(0)
    cam.release()
    return image


def transform_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cimg = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 2, 100, param1=50,
                               param2=30, minRadius=40, maxRadius=55)
    if circles is None:
        return False

    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        if(y <= 200):
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

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
