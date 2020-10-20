import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

class HWRModel:
    training_predictors = None
    test_predictors = None
    
    training_class = None
    test_class = None
    
    classifier = None
    
    
    def __init__(self):
        pass
    
    
    def load_data(self):
        (x_training, y_training), (x_test, y_test) = mnist.load_data()
        
        self.training_predictors = x_training.reshape(x_training.shape[0], 28, 28, 1)
        self.test_predictors = x_test.reshape(x_test.shape[0], 28, 28, 1)
        self.training_predictors = self.training_predictors.astype('float32')
        self.test_predictors = self.test_predictors.astype('float32')
        
        self.training_predictors /= 255
        self.test_predictors /= 255
        
        self.training_class = np_utils.to_categorical(y_training, 10)
        self.test_class = np_utils.to_categorical(y_test, 10)
    
    
    def create(self):
        # creating network
        self.classifier = Sequential()
        # first conv layer
        self.classifier.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(MaxPooling2D(pool_size=(2,2)))
        # second conv layer
        self.classifier.add(Conv2D(32, (3,3), activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(MaxPooling2D(pool_size=(2,2)))
        # flattern
        self.classifier.add(Flatten())
        # first dense layer
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dropout(0.2))
        # second dense layer
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dropout(0.2))
        # camada de saidas (units=10) ou seja digitos de 0 a 9
        self.classifier.add(Dense(units=10, activation='softmax'))
    
    
    def load(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.classifier = model_from_json(loaded_model_json)
    
    
    def save(self):
        model_json = self.classifier.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
    
    
    def compile(self):
        self.classifier.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    
    def train(self, times):
        self.classifier.fit(self.training_predictors, self.training_class,
                            batch_size = 128, epochs = times,
                            validation_data=(self.test_predictors, self.test_class))
    
    
    def get_accuracy(self):
        return self.classifier.evaluate(self.test_predictors, self.test_class)
