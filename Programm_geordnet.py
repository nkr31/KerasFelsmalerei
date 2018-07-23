# 3. Import libraries and modules

import numpy as np
import sys
import os.path
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from keras.preprocessing import image
import operator

WIDTH = 100
HEIGHT = 100

def create_model():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3), data_format='channels_last'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CATEGORIES, activation='softmax'))
    print("model erstellt")
    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("model compiliert")
    return model

def fit_model(model):
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=230)
    print("training abgeschlossen")
    return model

def evaluate_model(model):
    score = model.evaluate_generator(validation_generator)
    print(score)

def predict_picture(path):
    img = image.load_img(path, target_size=(WIDTH, HEIGHT))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]
    # check prediction
    pred = model.predict_classes(img_tensor)
    pred2 = model.predict_proba(img_tensor)

    # Index, repräsentiert Klassennr
    a=0
    wahrscheinlichkeiten = {}
    for i in pred2[0]:
        #erzeugt ein dictionary mit klassen als key und wahrscheinlichkeiten als value
        wahrscheinlichkeiten[CLASSES[a]] = i
        a=a+1
    #sortiert nach wahrscheinlichkeiten
    wahrscheinlichkeiten_sorted=sorted(wahrscheinlichkeiten.items(), key=operator.itemgetter(1) , reverse=True)
    print(wahrscheinlichkeiten_sorted)
    return pred[0]

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("model gespeichert")

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    return model

train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
print("train_datagen erzeugt")

test_datagen = ImageDataGenerator(rescale=1. / 255)
print("test_datagen erzeugt")

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical')

class_dictionary = train_generator.class_indices
CATEGORIES = len(class_dictionary)
CLASSES = {v: k for k, v in class_dictionary.items()}
print(CLASSES)
print("train_generator erzeugt")

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical')
print("validation_generator erzeugt")

#zum Laden:

if len(sys.argv) > 1:
    model = compile_model(load_model())
    print (CLASSES.get(predict_picture(sys.argv[1])))
elif os.path.isfile("model.json") and os.path.isfile("model.h5"):
    model = compile_model(load_model())
    evaluate_model(model)
    model = fit_model(model)
    evaluate_model(model)
    save_model(model)
else:
    model = create_model()
    model = compile_model(model)
    model = fit_model(model)
    evaluate_model(model)
    save_model(model)
