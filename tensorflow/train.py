import glob
import os


def get_all_image_training():
    dico = {}

    for filename in glob.iglob('images/**/*.png', recursive=True):
        directory_name = os.path.basename(os.path.dirname(filename))
        if directory_name not in dico:
            dico[directory_name] = []

        dico[directory_name].append(filename)
    return dico
