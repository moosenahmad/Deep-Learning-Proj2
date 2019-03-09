import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping
import numpy as np
import scipy.misc

mnist = tf.keras.datasets.mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

image_size = np.shape(x_train)[1]
n_channel = 1

x_train = x_train.reshape(x_train.shape[0], image_size, image_size, n_channel)
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, n_channel)

x_train_noise = x_train + 0.25 * np.random.normal(0.0, 1.0, x_train.shape)
x_test_noise = x_test + 0.25 * np.random.normal(0.0, 1.0, x_test.shape)

# Input of your model for training and testing
x_train_noise = np.clip(x_train_noise, 0.0, 1.0)
x_test_noise = np.clip(x_test_noise, 0.0, 1.0)

model = Sequential()
model.add(Conv2D(32, (4, 4), activation='relu', input_shape=(image_size, image_size, n_channel)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(UpSampling2D(size=2))
model.add(Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(UpSampling2D(size=2))
model.add(Conv2DTranspose(1, 1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', min_delta=0.01)
callback = [callback]

model.fit(x_train_noise, x_train, epochs=100, batch_size=36, callbacks=callback)

batch_size = 16
sample_idx = np.random.random_integers(low=0, high=10000, size=batch_size)
r_samples = x_test_noise[sample_idx, :, :, :]
rec_samples = model.predict(x=r_samples)
fig = np.zeros(shape=(image_size*2, image_size*batch_size, 1))
r_samples = r_samples * 255
rec_samples = rec_samples * 255

for idx in range(batch_size):
    fig[:image_size, idx*image_size: (idx+1)*image_size, :] = r_samples[idx, :, :]
    fig[image_size:, idx * image_size: (idx + 1) * image_size, :] = rec_samples[idx, :, :]

fig = fig.reshape(image_size*2, image_size*batch_size)
scipy.misc.imsave('out.jpg', fig)

