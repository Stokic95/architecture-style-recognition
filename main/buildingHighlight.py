import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers import  LSTM
from keras.optimizers import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time

def open_grayscale_image(_image_path):
    # otvara i vraca crno-bijelu sliku
    return cv2.imread(_image_path,0)

def make_histogram(_image_path):

    # kreira histogram koji moze da pokaze koliko je kojih piksela na slici
    # vrati listu od 256 jednoclanih lista, u svakoj listi je broj koji pokazuje koliko ima nekog piksela

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

    # normalizuje histogram slike
    # kao argument prima crno-bijelu sliku (ne putanju)
    # vraca crno-bijelu izjednacenu sliku

    # get image after histogram equalization
    _equalized_image = cv2.equalizeHist(_image)
    return _equalized_image

def show_images(_image_path):

    # 'stampa' nekoliko sliku na ekran radi uporedjivanja
    # argument je putanja do slike

    _original_image = cv2.imread(_image_path)
    _grayscale_image = cv2.imread(_image_path, 0)
    _equalized_image = equalize_histogram(_grayscale_image)
    _threshold_image1 = threshold_image(_grayscale_image)
    _threshold_image2 = threshold_image(_equalized_image)
    _canny_image1 = canny_edge_detection(_threshold_image1)
    _canny_image2 = canny_edge_detection(_threshold_image2)
    _canny_image3 = canny_image(_threshold_image2)

    #cv2.imshow("original", _original_image)
    #cv2.imshow("grayscale", _grayscale_image)
    #cv2.imshow("equalized", _equalized_image)
    #cv2.imshow("threshold", _threshold_image1)
    #cv2.imshow("threshold after equalization", _threshold_image2)
    #cv2.imshow("canny1", _canny_image1)
    #cv2.imshow("canny2", _canny_image2)
    #cv2.imshow("canny3", _canny_image3)
    #cv2.imshow("canny4", canny_image(threshold_image(_canny_image3)))
    cv2.waitKey(0)

def threshold_image(_image, _method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, _block_size=3, _constant=12):

    # odvaja zgradu od pozadine
    # argument je slika (ne putanja)
    # vraca obradjenu sliku

    # param 1 - image
    # param 3 - metod
    # param 5 - velicina bloka za pretragu (mora biti neparan broj veci od 1)
    # param 6 - parametar koji utice na pretragu, proizvoljan
    _threshold_image = cv2.adaptiveThreshold(_image, 255, _method, cv2.THRESH_BINARY, _block_size, _constant)
    return _threshold_image

def closing_image(_image):

    # treba da ocisti sliku
    # ne radi kako treba za sada

    _closing = cv2.morphologyEx(_image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    return _closing

def canny_edge_detection(_image):
    blur_image = cv2.GaussianBlur(_image, (5,5), 3)
    canny_image = cv2.Canny(blur_image, 1, 255)
    return canny_image

def canny_image(_image, _sigma=0.33):

    _v = np.median(_image)

    _lower = int(max(0, (1.0 - _sigma) * _v))
    _upper = int(min(255, (1.0 + _sigma) * _v))

    _canny_image = cv2.Canny(_image, _lower, _upper, True)
    return _canny_image

def resize_image(_image, _const):
    return cv2.resize(_image, _const, interpolation=cv2.INTER_NEAREST)

def convert_output(styles):
    outputs = [];
    for index in range(0, len(styles)):
        output = np.zeros(len(styles))
        output[index] = 1
        outputs.append(output)
    return outputs

def matrix_to_vector(_image):
    return _image.flatten()

def scale_to_range(_image):
    return _image/255

def prepare_for_ann(inputs):
    ready_for_ann = []
    for input in inputs:
        ready_for_ann.append(matrix_to_vector(scale_to_range(input)))
    return ready_for_ann

def create_ann(size):
    ann = Sequential()
    ann.add(LSTM(128, input_shape=(240,4900), activation ='sigmoid'))
    ann.add(Dense(128,activation='sigmoid'))
    ann.add(Dense(128,activation='sigmoid'))
    ann.add(Dense(3, activation = 'sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, shuffle=False)
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, style):
    result = []
    for output in outputs:
        result.append(style[winner(output)])
    return result

def load_data(path):
    data = []
    for i in os.listdir(path):
        data.append(i)
    return data

if __name__ == '__main__':

    start = time.time()

    gothic_path = "../data/training/gothic_training"
    modern_path = "../data/training/modern_training"
    renaissance_path = "../data/training/renaissance_training"
    gothic_test_path = "../data/training/gothic_test"
    modern_test_path = "../data/training/modern_test"
    renaissance_test_path = "../data/training/renaissance_test"
    gothic_data = load_data(gothic_path)
    modern_data = load_data(modern_path)
    renaissance_data = load_data(renaissance_path)
    gothic_test_data = load_data(gothic_test_path)
    modern_test_data = load_data(modern_test_path)
    renaissance_test_data = load_data(renaissance_test_path)


    ann_inputs = []
    ann_inputs_test = []
    inputs_gothic_train = []
    inputs_modern_train = []
    inputs_renaissance_train = []
    inputs_gothic_test = []
    inputs_modern_test = []
    inputs_renaissance_test = []

    styles = ['gothic', 'modern', 'renaissance']

    a = 0
    for path in gothic_data:
        _grayscale_image = open_grayscale_image(gothic_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_gothic_train.append(resize_image(_canny_image, (70,70)))

    for path in modern_data:
        _grayscale_image = open_grayscale_image(modern_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_modern_train.append(resize_image(_canny_image, (70, 70)))

    for path in renaissance_data:
        _grayscale_image = open_grayscale_image(renaissance_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_renaissance_train.append(resize_image(_canny_image, (70, 70)))


    ann_inputs_gothic_train = prepare_for_ann(inputs_gothic_train)
    ann_inputs_modern_train = prepare_for_ann(inputs_modern_train)
    ann_inputs_renaissance_train = prepare_for_ann(inputs_renaissance_train)
    ann_inputs.append(ann_inputs_gothic_train)
    ann_inputs.append(ann_inputs_modern_train)
    ann_inputs.append(ann_inputs_renaissance_train)

    ann_target_outputs = convert_output(styles)
    ann = create_ann(240)
   # ann.load_weights('traindata')
    ann = train_ann(ann, ann_inputs, ann_target_outputs)
    #result = ann.predict(np.array(ann_inputs, np.float32))
   # ann.save_weights('traindata')

    for path in gothic_test_data:
        _grayscale_image = open_grayscale_image(gothic_test_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_gothic_test.append(resize_image(_canny_image, (70,70)))

    for path in modern_test_data:
        _grayscale_image = open_grayscale_image(modern_test_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_modern_test.append(resize_image(_canny_image, (70, 70)))

    for path in renaissance_test_data:
        _grayscale_image = open_grayscale_image(renaissance_test_path + '/' + path)
        _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
        _canny_image = canny_edge_detection(_treshold_image)
        inputs_renaissance_test.append(resize_image(_canny_image, (70, 70)))


    ann_inputs_gothic_test = prepare_for_ann(inputs_gothic_test)
    ann_inputs_modern_test = prepare_for_ann(inputs_modern_test)
    ann_inputs_renaissance_test = prepare_for_ann(inputs_renaissance_test)
    ann_inputs_test.append(ann_inputs_gothic_test)
    ann_inputs_test.append(ann_inputs_modern_test)
    ann_inputs_test.append(ann_inputs_renaissance_test)




    annTest = Sequential()
    annTest.add(LSTM(128, input_shape=(60, 4900), activation='sigmoid'))
    annTest.add(Dense(128, activation='sigmoid'))
    annTest.add(Dense(128, activation='sigmoid'))
    annTest.add(Dense(3, activation='sigmoid'))
    annTest.set_weights(ann.get_weights())

    result = annTest.predict(np.array(ann_inputs_test, np.float32))
    print(display_result(result, styles))
    end = time.time()
    print(end - start)
    print(result)

    tri_slike = []
    tri_slike.append(ann_inputs_gothic_test[0])
    tri_slike.append(ann_inputs_modern_test[0])
    tri_slike.append(ann_inputs_renaissance_train[0])
    annTest1 = Sequential()
    annTest1.add(LSTM(128, input_shape=(1, 4900), activation='sigmoid'))
    annTest1.add(Dense(128, activation='sigmoid'))
    annTest1.add(Dense(128, activation='sigmoid'))
    annTest1.add(Dense(3, activation='sigmoid'))
    annTest1.set_weights(ann.get_weights())

    vrijednost = annTest1.predict(np.array(tri_slike), np.float32)
    print(vrijednost)


   #show_images(path)

    #image = open_grayscale_image(path)

    # equalize histogram of the image
    #equalized_image = equalize_histogram(image)

    # threshold image
    #threshold_image = threshold_image(image)

    #cv2.imshow("Equalized image", equalized_image)
    #cv2.imshow("Threshold image", threshold_image)
    #cv2.waitKey(0)
