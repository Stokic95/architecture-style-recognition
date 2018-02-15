from main.buildingHighlight import *
import os


def load_training_files(_style):
    # style - "gothic" | "modern" | "renaissance"
    _folder_path = "../data/training/" + _style + "_training"

    if (not os.path.exists(_folder_path)) or (not os.path.isdir(_folder_path)):
        print("Directory not found.")
        return []

    return os.listdir(_folder_path)


def prepare_image(_image_path):

    # otvori je crno-bijelu
    _image = open_grayscale_image(_image_path)

    # normalizuj histogram
    _image = equalize_histogram(_image)

    # izdvoj objekat thresholdingom
    _image = threshold_image(_image)

    # izdvoj ivice objekta canny operatorom
    _image = canny_edge_detection(_image)

    # smanji sliku
    _image = resize_image(_image)

    return _image


def resize_image(_image, _width=50, _height=50):
    _image = cv2.resize(_image, (_width, _height))
    return _image

if __name__ == '__main__':
    print("neuralNetwork.py ran..")