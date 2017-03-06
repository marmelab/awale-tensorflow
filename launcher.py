import argparse
from picture import picture


def process_image(save, numberPebble):
    image = picture.get_image_webcam()
    image = picture.transform_board(image, numberPebble, save)
    picture.display_picture(image)


def process_video():
    picture.transform_video()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("-s", "--save", action='store_true')
    ap.add_argument("-n", "--number")
    args = vars(ap.parse_args())
    if args["out"] == "image":
        process_image(args["save"], args["number"])
    elif args["out"] == "video":
        process_video()
