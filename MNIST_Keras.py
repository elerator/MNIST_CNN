import tensorflow as tf
import numpy as np
import os
import struct

from keras import backend as K


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import keras


class MNIST():
    def __init__(self, directory):
        self._directory = directory
        
        self._training_data = self._load_binaries("train-images.idx3-ubyte")
        self._training_labels = self._load_binaries("train-labels.idx1-ubyte")
        self._test_data = self._load_binaries("t10k-images.idx3-ubyte")
        self._test_labels = self._load_binaries("t10k-labels.idx1-ubyte")
        
        np.random.seed(0)
        samples_n = self._training_labels.shape[0]
        random_indices = np.random.choice(samples_n, samples_n // 10, replace = False)
        np.random.seed()
        
        self._validation_data = self._training_data[random_indices]
        self._validation_labels = self._training_labels[random_indices]
        self._training_data = np.delete(self._training_data, random_indices, axis = 0)
        self._training_labels = np.delete(self._training_labels, random_indices)
    
    def _load_binaries(self, file_name):
        path = os.path.join(self._directory, file_name)
        
        with open(path, 'rb') as fd:
            check, items_n = struct.unpack(">ii", fd.read(8))

            if "images" in file_name and check == 2051:
                height, width = struct.unpack(">II", fd.read(8))
                images = np.fromfile(fd, dtype = 'uint8')
                return np.reshape(images, (items_n, height, width))
            elif "labels" in file_name and check == 2049:
                return np.fromfile(fd, dtype = 'uint8')
            else:
                raise ValueError("Not a MNIST file: " + path)
    
    
    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)
    
    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)
    
    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)
    
    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]
        if batch_size <= 0:
            batch_size = samples_n
        
        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]
    
    
    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n
        
        
def init_model():
    K.clear_session()
    tf.reset_default_graph()

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding = "SAME", activation = "relu"))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    for layer in model.layers:
        print(layer.output_shape)
        
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(lr=0.1),
              metrics=['accuracy'])
              
    return model
        
        
def train(epochs = 1, batch_size=10):
    mnist = MNIST(os.getcwd())
    
    step = 1
    for epoch in range(epochs):

        print('Epoch = ', epoch + 1)
        prelim_acc = 0

        for batch, label_batch in mnist.get_training_batch(batch_size = 10):
            #label_batch = keras.utils.to_categorical(label_batch,10)
            stat = model.fit(np.expand_dims(batch,axis=3),label_batch, verbose = 0)#verbose =0 supresses output

            prelim_acc += stat.history["acc"][0]

            if (step % 100)==0:
                print(prelim_acc/100)
                prelim_acc = 0

            step+=1
            
    test_source, test_target = next(mnist.get_validation_batch(batch_size = 1000))
    model.evaluate(np.expand_dims(test_source,axis=3),test_target)
    

if __name__ == "__main__":
    model = init_model()
    train()

    



