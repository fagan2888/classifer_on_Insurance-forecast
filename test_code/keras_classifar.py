import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout
from keras.optimizers import RMSprop ,Adam
import matplotlib.pyplot as plt

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# Another way to build your neural net
model = Sequential([
    Dense(128, input_dim=784),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(128),
    Activation('relu'),
    Dropout(0.2),
    Dense(128) ,
    Activation('relu'),
    Dropout(0.2),### sigmoid ,relu , softmax , tanh
    Dense(10),
    Activation('softmax'),
        ])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(epsilon=1e-08)
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('Training ------------')
# Another way to train the model
his = model.fit(X_train, y_train,validation_split=0.25,epochs = 50, batch_size=400)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)


print('test loss: ', loss)
print('test accuracy: ', accuracy)



plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

