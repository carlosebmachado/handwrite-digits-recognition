from keras_preprocessing import image
from keras.models import model_from_json
import numpy as np


json_file = open('saved_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("saved_model/model.h5")


# COMPILE
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


# RECOGNIZE
test_img = image.load_img('test/digit-4_01.png', target_size=(28,28))
test_img = image.img_to_array(test_img)
test_img /= 255

test_img = np.expand_dims(test_img, axis=0)

test_img = test_img.reshape(3, 28, 28, 1)
test_img = test_img[0]
test_img = np.expand_dims(test_img, axis=0)

classifier.predict(test_img)
