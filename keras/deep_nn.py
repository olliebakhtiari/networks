# Python standard.
import time

# Third-party.
from keras.activations import relu
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np
from matplotlib import pyplot as plt

# Local.
from data.preprocessing import get_weather_10_min_interval, get_wind_energy

# tensorboard launch command - python -m tensorboard.main --logdir=~/logs --host=127.0.0.1
NAME = f'deep_nn_model-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

inputs = get_weather_10_min_interval()
outputs = get_wind_energy()

x_train, x_test, y_train, y_test = train_test_split(
    inputs,
    outputs,
    test_size=0.2,
)

model = Sequential()

model.add(Dense(512, input_shape=(6,), activation=relu, kernel_regularizer=l2(1e-7)))

model.add(Dropout(rate=0.05))
model.add(Dense(256, activation=relu, kernel_regularizer=l2(1e-7)))
model.add(Dropout(rate=0.05))
model.add(Dense(256, activation=relu, kernel_regularizer=l2(1e-7)))
model.add(Dropout(rate=0.05))
model.add(Dense(256, activation=relu, kernel_regularizer=l2(1e-7)))
model.add(Dropout(rate=0.05))
model.add(Dense(256, activation=relu, kernel_regularizer=l2(1e-7)))
model.add(Dropout(rate=0.05))

model.add(Dense(1))

adam_optimizer = Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=0.0,
    amsgrad=False,
)
model.compile(
    optimizer=adam_optimizer,
    loss='mean_squared_error',
    metrics=['mae', 'mse', 'mape'],
)
earlystopper = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=1,
    mode='auto',
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    min_delta=0.0001,
    cooldown=5,
    min_lr=10e-7,
)
history = model.fit(
    np.array(x_train),
    np.array(y_train),
    batch_size=32,
    epochs=500,
    validation_split=0.1,
    callbacks=[tensorboard, reduce_lr, earlystopper],
)
history_dict = history.history
val_loss = model.evaluate(np.array(x_test), np.array(y_test))

# plot values
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure()
plt.plot(loss_values, 'bo', label='training loss')
plt.plot(val_loss_values, 'r', label='val training loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')

y_train_pred = model.predict(np.array(x_train))
y_test_pred = model.predict(np.array(x_test))

print('Training dataset r2 score: {:0.3f}'.format(r2_score(y_train, y_train_pred)))
print('Testing dataset r2 score: {:0.3f}'.format(r2_score(y_test, y_test_pred)))

weights = model.get_weights()
print(weights)

plt.plot(y_train, y_train_pred, '*r')
plt.plot(y_test, y_test_pred, '*g')
plt.xlabel('Actual outputs')
plt.ylabel('Predicted outputs')

for i in range(0, 4500):
    plt.plot(i/100, i/100, '*b')