import argparse
import cv2
from picture import picture
from neural_network import training


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
    ap.add_argument("-t", "--train", action='store_true')
    ap.add_argument("-a", "--accuracy", action='store_true')
    ap.add_argument("-r", "--run", action='store_true')
    args = vars(ap.parse_args())
    if args["out"] == "image":
        number = "None" if args["number"] is None else args["number"]
        process_image(args["save"], number)
    elif args["out"] == "video":
        display_video()
    elif args["train"]:
        training.run_training()
    elif args["accuracy"]:
        training.display_accuracy()
    elif args["run"]:
        training.display_count_pebble()
