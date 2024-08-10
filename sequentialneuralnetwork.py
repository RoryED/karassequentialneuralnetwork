
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)

model = keras.Sequential( # sequential api
    [
        keras.Input(shape=(28*28,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

model = keras.Sequential()
model.add(keras.Input(shape=(28*28,)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(10))



# using functional api
#inputs = keras.Input(shape=(28, 28))
#x = layers.Dense(512, activation="relu")(inputs)
#x = layers.Dense(256, activation="relu")(x)
#outputs = layers.Dense(10, activation="softmax")(x)
#model = keras.Model(inputs=inputs, outputs=outputs)


model.compile( # specifies network configurations
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001), #sets the learning rate (using adam. other possibilities are gradient descent (momentum) Adagrad, RMSprop)
    metrics=["accuracy"], # the running accuracy so far
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2) # verbose means prints after each epochs, otherwise would be a progress bar
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
