from picture import picture


def process_image():
    image = picture.get_image_webcam()
    picture.transform_board(image)


if __name__ == "__main__":
    process_image()
