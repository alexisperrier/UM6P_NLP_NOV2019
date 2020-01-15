'''
MNIST
https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265
'''
import matplotlib.pyplot as plt
import tensorflow as tf
layers = tf.keras.layers
import numpy as np
print(tf.__version__)

# load data
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0



'''
Model definition
'''

# model = tf.keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

layer_dim = 64
epochs = 27

model = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(layer_dim),
    layers.Activation('relu'),
    layers.Dense(layer_dim),
    layers.Activation('relu'),
    layers.Dense(layer_dim),
    layers.Activation('relu'),
    layers.Dense(layer_dim),
    layers.Activation('relu'),
    layers.Dense(layer_dim),
    layers.Activation('relu'),
    layers.Dense(10),
    layers.Activation('softmax'),
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
Model fit
'''

res = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=2)
# print(model.summary())
'''
Model evaluation
'''

res_test  = model.evaluate(x_test, y_test, verbose = 0)
res_train = model.evaluate(x_train, y_train, verbose = 0);
print("test: {} train: {}".format( res_test, res_train ))

plt.plot(res.history['accuracy'], label = 'train 64x5')
plt.plot(res.history['val_accuracy'], label = 'test  64x5')
plt.grid(alpha = 0.3)
plt.legend()
plt.show()

if False:
    '''
    verification
    '''

    predictions = model.predict(x_test)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    predicted_label = class_names[np.argmax(predictions[0])]
    print('Actual label:', class_names[y_test[0]])
    print('Predicted label:', predicted_label)







# -----------------------------
