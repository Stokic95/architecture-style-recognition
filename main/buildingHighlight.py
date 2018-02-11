import cv2
import numpy as np


def open_grayscale_image(_image_path):
    return cv2.imread(_image_path, 0)

def make_histogram(_image_path):

    # open picture
    # param 1 - path to image
    # param 2 - 0 for grayscale
    _img = cv2.imread(_image_path, 0)

    # create histogram
    # param 1 - image
    # param 2 (wanted color) - [0] for grayscale
    # param 3 (mask) - None because we want full image
    # param 4 (BIN count) - [256] for full image
    # param 5 (intensity range) - [0, 256]
    _histogram = cv2.calcHist([_img], [0], None, [256], [0, 256])

    return _histogram


def equalize_histogram(_image):

    # get image after histogram equalization
    _equalized_image = cv2.equalizeHist(_image)

    return _equalized_image


def show_images(_image_path):

    _original_image = cv2.imread(_image_path)
    _grayscale_image = cv2.imread(_image_path, 0)
    _equalized_image = equalize_histogram(_grayscale_image)
    _threshold_image1 = threshold_image(_grayscale_image)
    _threshold_image2 = threshold_image(_equalized_image)

    cv2.imshow("original", _original_image)
    cv2.imshow("grayscale", _grayscale_image)
    cv2.imshow("equalized", _equalized_image)
    cv2.imshow("threshold", _threshold_image1)
    cv2.imshow("threshold after equalization", _threshold_image2)
    cv2.waitKey(0)

def threshold_image(_image):

    _threshold_image = cv2.adaptiveThreshold(_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 12)
    return _threshold_image

def closing_image(_image):

    _closing = cv2.morphologyEx(_image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    return _closing

if __name__ == '__main__':

    path = "../data/training/gothic_training/baptistry1.jpg"

    show_images(path)

    image = open_grayscale_image(path)

    # equalize histogram of the image
    #equalized_image = equalize_histogram(image)

    # threshold image
    #threshold_image = threshold_image(image)

    #cv2.imshow("Equalized image", equalized_image)
    #cv2.imshow("Threshold image", threshold_image)
    #cv2.waitKey(0)
