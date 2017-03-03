import argparse
from picture import picture


def process_image():
    image = picture.get_image_webcam()
    image = picture.transform_board(image)
    picture.display_picture(image)


def process_video():
    picture.transform_video()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    args = vars(ap.parse_args())
    if args["out"] == "image":
        process_image()
    elif args["out"] == "video":
        process_video()
