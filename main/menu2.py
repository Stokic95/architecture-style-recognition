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
            pass
            #resize_image(_path)
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

def prepare_image(_path, _size):
    _img = cv2.imread(_path)
    _a = []
    _a.append(_img)

    _augmanted_images = augment_data(np.array(_a))

    _result = []

    for _image in _augmanted_images:
        _image = cv2.cvtColor(_image,cv2.COLOR_BGR2GRAY)
        _image = equalize_histogram(_image)
        _image = threshold_image(_image)
        _image = canny_edge_detection(_image)
        _image = resize_image(_image, (_size, _size))

        parts = []

        meisure = int(_size/3)

        parts.append(_image[0:meisure, 0:meisure])
        parts.append(_image[meisure:meisure*2, 0:meisure])
        parts.append(_image[meisure*2:_size, 0:meisure])
        parts.append(_image[0:meisure, meisure:meisure*2])
        parts.append(_image[meisure:meisure*2, meisure:meisure*2])
        parts.append(_image[meisure*2:_size, meisure:meisure*2])
        parts.append(_image[0:meisure, meisure*2:_size])
        parts.append(_image[meisure:meisure*2, meisure*2:_size])
        parts.append(_image[meisure*2:_size, meisure*2:_size])

        for part in parts:
            if sum(scale_to_range(matrix_to_vector(part))) > (_size*_size)/45:
                _result.append(part)

    return _result

def prepare_inputs():
    _gothic_training_input_files = load_data("../data/training/gothic_training")
    _modern_training_input_files = load_data("../data/training/modern_training")
    _renaissance_training_input_files = load_data("../data/training/renaissance_training")

    _gothic_training_inputs = []
    _modern_training_inputs = []
    _renaissance_training_inputs = []

    for name in _gothic_training_input_files:
        i = 0
        print("preparing training gothic: " + str(name))
        for part in prepare_image("../data/training/gothic_training/" + name, 180):
            i = i + 1
            _gothic_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/training/network_inputs/gothic/" + name.split('.')[0] + str(i) + ".jpg", part)

    for name in _modern_training_input_files:
        i = 0
        print("preparing training modern: " + str(name))
        for part in prepare_image("../data/training/modern_training/" + name, 180):
            i = i + 1
            _modern_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/training/network_inputs/modern/" + name.split('.')[0] + str(i) + ".jpg", part)

    for name in _renaissance_training_input_files:
        i = 0
        print("preparing training renaissance: " + str(name))
        for part in prepare_image("../data/training/renaissance_training/" + name, 180):
            i = i + 1
            _renaissance_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/training/network_inputs/renaissance/" + name.split('.')[0] + str(i) + ".jpg", part)

    sizes = []
    sizes.append(len(_gothic_training_inputs))
    sizes.append(len(_modern_training_inputs))
    sizes.append(len(_renaissance_training_inputs))

    minimal = min(sizes)

    _inputs = []
    _inputs.append(_gothic_training_inputs[:minimal])
    _inputs.append(_modern_training_inputs[:minimal])
    _inputs.append(_renaissance_training_inputs[:minimal])

    print("training minimal: " + str(minimal))

    return _inputs, minimal

def prepare_test_inputs():
    _gothic_training_input_files = load_data("../data/test/gothic_test")
    _modern_training_input_files = load_data("../data/test/modern_test")
    _renaissance_training_input_files = load_data("../data/test/renaissance_test")

    _gothic_training_inputs = []
    _modern_training_inputs = []
    _renaissance_training_inputs = []

    for name in _gothic_training_input_files:
        i = 0
        print("preparing test gothic: " + str(name))
        for part in prepare_image("../data/test/gothic_test/" + name, 180):
            i = i + 1
            _gothic_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/test/network_inputs/gothic/" + name.split('.')[0] + str(i) + ".jpg", part)

    for name in _modern_training_input_files:
        i = 0
        print("preparing test modern: " + str(name))
        for part in prepare_image("../data/test/modern_test/" + name, 180):
            i = i + 1
            _modern_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/test/network_inputs/modern/" + name.split('.')[0] + str(i) + ".jpg", part)

    for name in _renaissance_training_input_files:
        i = 0
        print("preparing test renaissance: " + str(name))
        for part in prepare_image("../data/test/renaissance_test/" + name, 180):
            i = i + 1
            _renaissance_training_inputs.append(scale_to_range(matrix_to_vector(part)))
            cv2.imwrite("../data/test/network_inputs/renaissance/" + name.split('.')[0] + str(i) + ".jpg", part)

    sizes = []
    sizes.append(len(_gothic_training_inputs))
    sizes.append(len(_modern_training_inputs))
    sizes.append(len(_renaissance_training_inputs))

    minimal = min(sizes)

    _inputs = []
    _inputs.append(_gothic_training_inputs[:minimal])
    _inputs.append(_modern_training_inputs[:minimal])
    _inputs.append(_renaissance_training_inputs[:minimal])

    print("test minimal: " + str(minimal))

    return _inputs, minimal

def augment_data(dataset, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	augmented_image = []

	for num in range (0, dataset.shape[0]):

		for i in range(0, augementation_factor):
			# original image:
			augmented_image.append(dataset[num])

			if use_random_rotation:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_shear:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_shift:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))

			if use_random_zoom:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.7, 0.7), row_axis=0, col_axis=1, channel_axis=2))

	return np.array(augmented_image)

def network_training_inputs(_size):
    _gothic_training_files = load_data("../data/training/network_inputs/gothic")
    _modern_training_files = load_data("../data/training/network_inputs/modern")
    _renaiassance_training_files = load_data("../data/training/network_inputs/renaissance")

    _inputs = []

    for _name in _gothic_training_files:
        _im = cv2.imread("../data/training/network_inputs/gothic/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    for _name in _modern_training_files:
        _im = cv2.imread("../data/training/network_inputs/modern/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    for _name in _renaiassance_training_files:
        _im = cv2.imread("../data/training/network_inputs/renaissance/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    _inputs = np.array(_inputs).reshape(-1, _size, _size, 1)
    return _inputs

def network_test_inputs(_size):
    _gothic_test_files = load_data("../data/test/network_inputs/gothic")
    _modern_test_files = load_data("../data/test/network_inputs/modern")
    _renaiassance_test_files = load_data("../data/test/network_inputs/renaissance")

    _inputs = []

    for _name in _gothic_test_files:
        _im = cv2.imread("../data/test/network_inputs/gothic/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    for _name in _modern_test_files:
        _im = cv2.imread("../data/test/network_inputs/modern/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    for _name in _renaiassance_test_files:
        _im = cv2.imread("../data/test/network_inputs/renaissance/" + _name)
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _inputs.append(scale_to_range(_im))

    _inputs = np.array(_inputs).reshape(-1, _size, _size, 1)
    return _inputs

if __name__ == '__main__':
    main_menu()