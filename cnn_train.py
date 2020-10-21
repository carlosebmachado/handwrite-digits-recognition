from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# DATA LOAD
(x_training, y_training), (x_test, y_test) = mnist.load_data()

training_predictors = x_training.reshape(x_training.shape[0], 28, 28, 1)
test_predictors = x_test.reshape(x_test.shape[0], 28, 28, 1)
training_predictors = training_predictors.astype('float32')
test_predictors = test_predictors.astype('float32')

training_predictors /= 255
test_predictors /= 255

training_class = np_utils.to_categorical(y_training, 10)
test_class = np_utils.to_categorical(y_test, 10)


# CNN CREATE
classifier = Sequential()

# first conv layer
classifier.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
# second conv layer
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
# flattern
classifier.add(Flatten())

# first dense layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
# second dense layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
# camada de saidas (units=10) ou seja digitos de 0 a 9
classifier.add(Dense(units=10, activation='softmax'))


# COMPILE
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


# TRAIN
classifier.fit(training_predictors, training_class,
               batch_size = 128, epochs = 5,
               validation_data=(test_predictors, test_class))


# SAVE CNN
model_json = classifier.to_json()
with open("saved_model/model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("saved_model/model.h5")


# EVALUATE
classifier.evaluate(test_predictors, test_class)
