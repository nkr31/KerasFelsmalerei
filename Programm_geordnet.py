import numpy as np
import sys
import os.path
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from keras.preprocessing import image
import operator

WIDTH = 100
HEIGHT = 100

# erstellt ein Model
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

# compiliert das Model
def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("model compiliert")
    return model

# trainiert das Model
def fit_model(model):
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=230)
    print("training abgeschlossen")
    return model

# evaluiert das Model
def evaluate_model(model):
    score = model.evaluate_generator(validation_generator)
    print(score)

# ermöglicht die Klassifizierung eines einzelnen Bildes
def predict_picture(path):
    img = image.load_img(path, target_size=(WIDTH, HEIGHT))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.
    # check prediction
    pred = model.predict_classes(img_tensor)
    pred2 = model.predict_proba(img_tensor)

    # Index, repräsentiert Klassennr
    a=0
    wahrscheinlichkeiten = {}
    for i in pred2[0]:
        #erzeugt ein Dictionary mit Klassen als Key und Wahrscheinlichkeiten als Value
        wahrscheinlichkeiten[CLASSES[a]] = i
        a=a+1
    # Tierklassen nach Wahrscheinlichkeiten sortiert
    wahrscheinlichkeiten_sorted=sorted(wahrscheinlichkeiten.items(), key=operator.itemgetter(1) , reverse=True)
    print(wahrscheinlichkeiten_sorted)
    return pred[0]

# Model speichern für einen späteren Zeitpunkt
def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("model gespeichert")

# bereits vorhandenes Model aus .h5- und .json-Datei laden
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    return model

# Vorbereitung der Daten
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
# Anzahl der gefundenen Tierklassen
CATEGORIES = len(class_dictionary)
# Dictionary mit Tierklassen und deren Indizes
CLASSES = {v: k for k, v in class_dictionary.items()}
print(CLASSES)
print("train_generator erzeugt")

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical')
print("validation_generator erzeugt")

# zum Laden:
# wenn ein zweiter Parameter übergeben wird, wird das Bild am entsprechenden Pfad klassifiziert
if len(sys.argv) > 1:
    model = compile_model(load_model())
    evaluate_model(model)
    print (CLASSES.get(predict_picture(sys.argv[1])))
# wenn bereits Speicherdateien vorhanden sind, aber kein zweiter Parameter übergeben wird, wird das geladene Model neu trainiert und gespeichert
elif os.path.isfile("model.json") and os.path.isfile("model.h5"):
    model = compile_model(load_model())
    evaluate_model(model)
    model = fit_model(model)
    evaluate_model(model)
    save_model(model)
# Default: erstellt, compiliert, trainiert, evaluiert und speichert das Model
else:
    model = create_model()
    model = compile_model(model)
    model = fit_model(model)
    evaluate_model(model)
    save_model(model)
