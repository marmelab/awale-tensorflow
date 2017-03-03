import argparse
from picture import picture


def process_image(save):
    image = picture.get_image_webcam()
    image = picture.transform_board(image, save)
    picture.display_picture(image)


def process_video():
    picture.transform_video()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("-s", "--save", action='store_true')
    args = vars(ap.parse_args())
    if args["out"] == "image":
        process_image(args["save"])
    elif args["out"] == "video":
        process_video()
