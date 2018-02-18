import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from main.menu2 import *
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

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

def convert_output(_size):
    meisure = int(_size / 3)
    outputs = [];
    gothic = [1, 0, 0]
    modern = [0, 1, 0]
    renaissance = [0 , 0 , 1]
    for index in range(0, meisure):
        outputs.append(gothic)
    for index in range(meisure, meisure*2):
        outputs.append(modern)
    for index in range(meisure*2 ,_size):
        outputs.append(renaissance)
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

def create_ann(_size):
    ann = Sequential()
    ann.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(_size, _size,1), padding='same'))
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
    ann.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
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
    ann_inputs = network_training_inputs(80)
    ann_target_outputs = convert_output(12846)

    # kreiraj neuronsku mrezu
    print("creating ann..")
    ann = create_ann(80)
    print("ann created")

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

    '''
        Treniranje
    '''
   # ann = training()

    '''
        Testiranje
    '''
    ann_inputs_test = network_test_inputs(80)

    print("creating test ann..")
    annTest = create_ann(80)
    print("test ann created")

    # postavi tezine mreze za treniranje
    annTest.load_weights("weights")

    # predvidi izlaz za poslati ulaz
    print("started prediction..")
    result = annTest.predict(np.array(ann_inputs_test, np.float32))
    print("prediction completed")

    # stampaj rezultat
    result1 = display_result(result, styles)
    print("Neural network results:")
    print()
    gothic1 = 0
    modern1 = 0
    renaissance1 = 0

    for i in range(0,850):
        if result1[i] == "gothic":
            gothic1 = gothic1 + 1
        elif result1[i] == "modern":
            modern1 = modern1 + 1
        else:
            renaissance1 = renaissance1 + 1
    print("Gothic_Test_1:  " + str((gothic1/850.0)*100) + "%")
    print("Modern_Test_1:  " + str((modern1 / 850.0)*100) + "%")
    print("Renaissance_Test_1:  " + str((renaissance1 / 850.0)*100) + "%")

    gothic1 = 0
    modern1 = 0
    renaissance1 = 0

    for i in range(850, 1700):
        if result1[i] == "gothic":
            gothic1 = gothic1 + 1
        elif result1[i] == "modern":
            modern1 = modern1 + 1
        else:
            renaissance1 = renaissance1 + 1
    print('---------------------------------------------------')
    print("Gothic_Test_2:  " + str((gothic1 / 850.0)*100) + "%")
    print("Modern_Test_2: " + str((modern1 / 850.0)*100) + "%")
    print("Renaissance_Test_2:  " + str((renaissance1 / 850.0)*100) + "%")

    gothic1 = 0
    modern1 = 0
    renaissance1 = 0

    for i in range(1700, 2550):
        if result1[i] == "gothic":
            gothic1 = gothic1 + 1
        elif result1[i] == "modern":
            modern1 = modern1 + 1
        else:
            renaissance1 = renaissance1 + 1
    print('---------------------------------------------------')
    print("Gothic_Test_3: " + str((gothic1 / 850.0)*100) + "%")
    print("Modern_Test_3: " + str((modern1 / 850.0)*100) + "%")
    print("Renaissance_Test_3: " + str((renaissance1 / 850.0)*100) + "%")

    # istampaj vrijeme
    end = time.time()
    print(end - start)
