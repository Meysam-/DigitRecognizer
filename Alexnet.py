from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd

from sklearn import preprocessing


class Alexnet:
    def __init__(self):
        np.random.seed(1000)

        self.model = self.create_model()
        self.compile_model()

    def create_model(self):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(28, 28, 1), kernel_size=(5, 5), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.summary()

        return model

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, Y_train):
        self.model.fit(self.X_train, self.Y_train, batch_size=64, epochs=1, verbose=1,
                       validation_split=0.2, shuffle=True)

    def predict(self, X_test):
        return self.model.predict_classes(X_test, verbose=1)

    def save(self):
        self.model.save_weights("weights/alexnet-weights.h5")

    def load(self):
        self.model.load_weights("weights/alexnet-weights.h5")


def read_data():
    train = pd.read_csv("train.csv").values
    test = pd.read_csv("test.csv").values
    return train, test


if __name__ == "__main__":
    train, test = read_data()

    trainX = train[:, 1:].reshape(train.shape[0], 28, 28, 1).astype('float32')
    X_train = trainX / 255.0
    Y_train = train[:, 0]

    lb = preprocessing.LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)

    X_test = test.reshape(test.shape[0], 28, 28, 1)

    alexnet = Alexnet()
    # alexnet.train(X_train, Y_train)

    alexnet.load()

    preds = alexnet.predict(X_test)
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds})\
        .to_csv("predicts.cvs", index=False, header=True)

