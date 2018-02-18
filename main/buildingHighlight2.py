import cv2
import os
import numpy as np
import  keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers import  LSTM
from keras.optimizers import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
from main.menu import *
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


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
    prvi = [1, 0, 0]
    drugi = [0, 1, 0]
    treci = [0 , 0 , 1]
    for index in range(0, 4282):
        outputs.append(prvi)
    for index in range(4282, 8564):
        outputs.append(drugi)
    for index in range(8564,12846):
        outputs.append(treci)
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

def create_ann(_size, _pixels):
    ann = Sequential()
    ann.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(80, 80,1), padding='same'))
    ann.add(keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    ann.add(MaxPooling2D((2, 2), padding='same'))
    ann.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    ann.add(keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    ann.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    ann.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    ann.add(keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    ann.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    ann.add(Flatten())
    ann.add(Dense(128, activation='linear'))
    ann.add(keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    ann.add(Dense(3, activation='softmax'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    print(X_train.shape)
    y_train = np.array(y_train, np.float32)
    ann.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
    ann.summary()
    ann.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1)

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


def training():

    # # putanje
    # gothic_training_path = "../data/training/gothic_training"
    # modern_training_path = "../data/training/modern_training"
    # renaissance_training_path = "../data/training/renaissance_training"
    #
    # # ucitaj trening fajlove
    # gothic_training_data = load_data(gothic_training_path)
    # modern_training_data = load_data(modern_training_path)
    # renaissance_training_data = load_data(renaissance_training_path)
    #
    # ann_inputs = []  # ulazni vektor za neuronsku mrezu
    # inputs_gothic_train = []  # obradjene trening slike gotike
    # inputs_modern_train = []  # obradjene trening slike moderne
    # inputs_renaissance_train = []  # obradjene trening slike renesanse
    #
    # # ucitaj svaku sliku za treniranje gotike
    # for path in gothic_training_data:
    #     print("preparing for gothic training: " + path)
    #     _grayscale_image = open_grayscale_image(gothic_training_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_gothic_train.append(resize_image(_canny_image, (70, 70)))
    #
    # # ucitaj svaku sliku za treniranje moderne
    # for path in modern_training_data:
    #     print("preparing for modern training: " + path)
    #     _grayscale_image = open_grayscale_image(modern_training_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_modern_train.append(resize_image(_canny_image, (70, 70)))
    #
    # # ucitaj svaku sliku za treniranje renesanse
    # for path in renaissance_training_data:
    #     print("preparing for renaissance training: " + path)
    #     _grayscale_image = open_grayscale_image(renaissance_training_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_renaissance_train.append(resize_image(_canny_image, (70, 70)))
    #
    # # pripremi trening podatke za ulaz u mrezu
    # print("preparing training inputs for ann..")
    # ann_inputs_gothic_train = prepare_for_ann(inputs_gothic_train)
    # ann_inputs_modern_train = prepare_for_ann(inputs_modern_train)
    # ann_inputs_renaissance_train = prepare_for_ann(inputs_renaissance_train)
    #
    # # dodaj sva tri ulaza u jedan vektor
    # ann_inputs.append(ann_inputs_gothic_train)
    # ann_inputs.append(ann_inputs_modern_train)
    # ann_inputs.append(ann_inputs_renaissance_train)

    ann_inputs = network_training_inputs()

    # trenirani izlaz, vraca: [ [100], [010], [001] ]
    ann_target_outputs = convert_output(styles)

    # kreiraj neuronsku mrezu sa 240 ulaza
    print("creating ann..")
    ann = create_ann(4282, 6400)
    print("ann created")

    # ucitaj sacuvane tezine
    print("loading weights..")
    #ann.load_weights("weights")
    print("weights loaded")

    # treniraj mrezu
    print("started training..")
    ann = train_ann(ann, ann_inputs, ann_target_outputs)
    print("training completed")

    # sacuvaj tezine
    print("saving weights to file..")
    ann.save_weights("weights")
    print("weights saved to file")

    return ann


if __name__ == '__main__':

    styles = ['gothic', 'modern', 'renaissance']

    # pocni da brojis vrijeme
    start = time.time()

    gothic_test_path = "../data/test/gothic_test"
    modern_test_path = "../data/test/modern_test"
    renaissance_test_path = "../data/test/renaissance_test"

    '''
        Treniranje
    '''

   # ann = training()
    #ann = create_ann(240)
    #ann.load_weights("weights")

    '''
        Testiranje
    '''

    # ann_inputs_test = [] # ulazni vektor u mrezu za testiranje
    # inputs_gothic_test = [] # obradjene test slike za gotiku
    # inputs_modern_test = [] # obradjene test slike za modernu
    # inputs_renaissance_test = [] # obradjene test slike za renesansu
    #
    # # ucitaj test fajlove
    # gothic_test_data = load_data(gothic_test_path)
    # modern_test_data = load_data(modern_test_path)
    # renaissance_test_data = load_data(renaissance_test_path)
    #
    # # obradi slike za gotiku
    # for path in gothic_test_data:
    #     print("preparing for gothic test: " + path)
    #     _grayscale_image = open_grayscale_image(gothic_test_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_gothic_test.append(resize_image(_canny_image, (60,60)))
    #
    # # obradi slike za modernu
    # for path in modern_test_data:
    #     print("preparing for modern test: " + path)
    #     _grayscale_image = open_grayscale_image(modern_test_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_modern_test.append(resize_image(_canny_image, (60, 60)))
    #
    # # obradi slike za renesansu
    # for path in renaissance_test_data:
    #     print("preparing for renaissance test: " + path)
    #     _grayscale_image = open_grayscale_image(renaissance_test_path + '/' + path)
    #     _treshold_image = threshold_image(equalize_histogram(_grayscale_image))
    #     _canny_image = canny_edge_detection(_treshold_image)
    #     inputs_renaissance_test.append(resize_image(_canny_image, (60, 60)))
    #
    # # pripremi ulaze za mrezu
    # print("preparing test inputs for ann..")
    # ann_inputs_gothic_test = prepare_for_ann(inputs_gothic_test)
    # ann_inputs_modern_test = prepare_for_ann(inputs_modern_test)
    # ann_inputs_renaissance_test = prepare_for_ann(inputs_renaissance_test)
    #
    # # smjesti ulaze u jedan vektor
    # ann_inputs_test.append(ann_inputs_gothic_test)
    # ann_inputs_test.append(ann_inputs_modern_test)
    # ann_inputs_test.append(ann_inputs_renaissance_test)

    ann_inputs_test = network_test_inputs()

    # kreiraj mrezu za testiranje sa 60 ulaza
    print("creating test ann..")
    annTest = create_ann(850, 6400)
    print("test ann created")

    # postavi tezine mreze za treniranje
    #annTest.set_weights(ann.get_weights())
    annTest.load_weights("weights")
    # predvidi izlaz za poslati ulaz
    print("started prediction..")
    result = annTest.predict(np.array(ann_inputs_test, np.float32))
    print("prediction completed")

    # stampaj rezultat
    rezultat = display_result(result, styles)

    gotika1 = 0
    moderna1 = 0
    renesansa1 = 0

    for i in range(0,850):
        if rezultat[i] == "gothic":
            gotika1 = gotika1 + 1
        elif rezultat[i] == "modern":
            moderna1 = moderna1 + 1
        else:
            renesansa1 = renesansa1 + 1
    print("Gotika1:  " + str(gotika1/850.0))
    print("Moderna1:  " + str(moderna1 / 850.0))
    print("Renesansa1:  " + str(renesansa1 / 850.0))

    gotika1 = 0
    moderna1 = 0
    renesansa1 = 0

    for i in range(850, 1700):
        if rezultat[i] == "gothic":
            gotika1 = gotika1 + 1
        elif rezultat[i] == "modern":
            moderna1 = moderna1 + 1
        else:
            renesansa1 = renesansa1 + 1
    print("Gotika2:  " + str(gotika1 / 850.0))
    print("Moderna2: " + str(moderna1 / 850.0))
    print("Renesansa2:  " + str(renesansa1 / 850.0))

    gotika1 = 0
    moderna1 = 0
    renesansa1 = 0

    for i in range(1700, 2550):
        if rezultat[i] == "gothic":
            gotika1 = gotika1 + 1
        elif rezultat[i] == "modern":
            moderna1 = moderna1 + 1
        else:
            renesansa1 = renesansa1 + 1
    print("Gotika3: " + str(gotika1 / 850.0))
    print("Moderna3: " + str(moderna1 / 850.0))
    print("Renesansa3: " + str(renesansa1 / 850.0))




    # istampaj vrijeme
    end = time.time()
    print(end - start)

    # pokazi rezultat
    print(result)
