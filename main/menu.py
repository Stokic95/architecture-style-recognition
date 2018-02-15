from main.buildingHighlight import *
import os.path


def main_menu():
    print("1 - process picture")

    _option = input("Option: ")

    if _option == "1":
        _path = "../data/training/" + input("Picture path: ")

        if not os.path.isfile(_path):
            print("File not found")
            main_menu()

        process_picture(_path)


def process_picture(_path):

        print("1 - show original")
        print("2 - show grayscale")
        print("3 - show grayscale after histogram equalization")
        print("4 - show grayscale and grayscale after histogram equalization")
        print("5 - show threshold image")
        print("6 - show canny image")
        print("7 - show canny on blur image")
        print("8 - show canny and canny on blur image")
        print("9 - resize image")
        print("0 - back")

        _option = input("Option: ")

        if _option == "1":
            show_original(_path)
        elif _option == "2":
            show_grayscale(_path)
        elif _option == "3":
            show_grayscale_after_histogram_equalization(_path)
        elif _option == "4":
            show_two_grayscales(_path)
        elif _option == "5":
            show_threshold_image(_path)
        elif _option == "6":
            show_canny_image(_path)
        elif _option == "7":
            show_canny_on_blur_image(_path)
        elif _option == "8":
            show_two_cannies(_path)
        elif _option == "9":
            resize_image(_path)
        elif _option == "0":
            main_menu()
        else:
            process_picture(_path)


def show_original(_path):
    cv2.imshow("Original", cv2.imread(_path))
    cv2.waitKey(0)
    process_picture(_path)


def show_grayscale(_path):
    cv2.imshow("Grayscale", open_grayscale_image(_path))
    cv2.waitKey(0)
    process_picture(_path)


def show_grayscale_after_histogram_equalization(_path):
    cv2.imshow("Grayscale after histogram equalization", equalize_histogram(open_grayscale_image(_path)))
    cv2.waitKey(0)
    process_picture(_path)


def show_threshold_image(_path):
    _option = input("Method (0 for GAUSSIAN, 1 for MEAN)(GAUSSIAN is default): ")
    _method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    if _option == "1":
        _method = cv2.ADAPTIVE_THRESH_MEAN_C

    _block_size = 3
    _option = input("Block size (default is 3): ")
    try:
        _block_size = int(_option)
    except:
        pass

    _constant = 12
    _option = input("Constant (default is 12): ")
    try:
        _constant = int(_option)
    except:
        pass

    cv2.imshow("Threshold image", threshold_image(equalize_histogram(open_grayscale_image(_path)), _method, _block_size, _constant))
    cv2.waitKey(0)
    process_picture(_path)


def show_canny_image(_path):
    _sigma = 0.33
    _option = input("Sigma (default is 0.33): ")
    try:
        _sigma = float(_option)
    except:
        pass

    cv2.imshow("Canny image", canny_image(threshold_image(equalize_histogram(open_grayscale_image(_path))), _sigma))
    cv2.waitKey(0)
    process_picture(_path)


def show_canny_on_blur_image(_path):
    cv2.imshow("Canny on blur image", canny_edge_detection(threshold_image(equalize_histogram(
        open_grayscale_image(_path)))))
    cv2.waitKey(0)
    process_picture(_path)


def show_two_cannies(_path):
    _image = threshold_image(equalize_histogram(open_grayscale_image(_path)))
    cv2.imshow("Canny and canny on blur image", np.hstack((canny_image(_image), canny_edge_detection(_image))))
    cv2.waitKey(0)
    process_picture(_path)


def show_two_grayscales(_path):
    _image = open_grayscale_image(_path)
    cv2.imshow("Grayscale and grayscale after histogram equalization", np.hstack((_image, equalize_histogram(_image))))
    cv2.waitKey(0)
    process_picture(_path)


def resize_image(_path):
    _image = open_grayscale_image(_path)
    _image = cv2.resize(_image, (50, 50))
    cv2.imshow("resized", _image)
    cv2.waitKey(0)
    process_picture(_path)

if __name__ == '__main__':
    main_menu()
