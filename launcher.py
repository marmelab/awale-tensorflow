import argparse
from picture import picture
import cv2


def process_image(save, number_pebble):
    image = picture.get_image_webcam()
    image = picture.detect_and_crop_pit(image, number_pebble, save)
    picture.display_picture(image, "Output")
    cv2.waitKey(0)


def display_video():
    picture.detect_and_crop_pit_video()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("-s", "--save", action='store_true')
    ap.add_argument("-n", "--number")
    args = vars(ap.parse_args())
    if args["out"] == "image":
        process_image(args["save"], args["number"])
    elif args["out"] == "video":
        display_video()
