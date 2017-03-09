import glob
import os


def get_all_image_training():
    dico = {}

    for filename in glob.iglob('images/**/*.png', recursive=True):
        name = os.path.basename(filename)
        directory_name = os.path.basename(os.path.dirname(filename))

        if directory_name not in dico:
            dico[directory_name] = []

        dico[directory_name].append(name)
    return dico
