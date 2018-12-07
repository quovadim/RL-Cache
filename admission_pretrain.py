import pandas as pd
import numpy as np

import keras.layers as l
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

#frame = pd.read_csv('res_128_0', delimiter=' ', names=['timestamp', 'id', 'size', 'admit'], index_col=False)
admission = np.load('auxiliary/admissions_optimal_pfoo.npy')

indicies = [i for i in range(len(admission)) if admission[i] != -1]
#data = np.asarray(np.load('features_secondary.npy'))
data_file = open('auxiliary/features_secondary.npy', 'r')
data = []
counter = 0
for line in tqdm(data_file, total=len(admission)):
    if counter == len(admission):
        break
    if admission[counter] != -1:
        df = line.split(' ')
        df = [int(item) for item in df]
        data.append(np.asarray(df))
    counter += 1

admission = admission[indicies]

data = np.asarray(data)

fnum = 299

print data.shape

X = data[:, :fnum]
timestamps = data[:, fnum]
ids = data[:, fnum + 1]
size = data[:, fnum + 2]

opt_size = min(len(X), len(admission))

#admission = frame['admit'].values

X = X[:opt_size]
y = admission[:opt_size]

print X.shape, y.shape

opt_size = len(y)

y_converted = []
for i in range(opt_size):
    y_t = np.zeros(2)
    y_t[y[i]] = 1
    y_converted.append(y_t)
y = np.asarray(y_converted)

X = X[100000:, :]
y = y[100000:, :]

dropout_rate = 0.2

model_admission = Sequential()
dim = fnum
multiplier = 2
layers = 5
parameter = 0.1
# self.model_admission.add(l.BatchNormalization(input_shape=(featurer.dim,)))
model_admission.add(l.Dense(dim * multiplier, input_shape=(dim,), activation='elu'))
#model_admission.add(l.BatchNormalization())
for _ in range(layers):
    model_admission.add(l.Dropout(dropout_rate))
    model_admission.add(l.Dense(dim * multiplier, activation='elu'))
model_admission.add(l.Dropout(dropout_rate))
model_admission.add(l.Dense(2, activation='softmax'))

adm_optimizezr = Adam(lr=1e-4)

callbacks = [ModelCheckpoint('models/adm_model', save_weights_only=True, verbose=1)]

model_admission.compile(adm_optimizezr, loss='categorical_crossentropy', metrics=['accuracy'])

model_admission.summary()

try:
    model_admission.fit(X, y, batch_size=2048 * 4, epochs=50, shuffle=True, validation_split=0.1, callbacks=callbacks)
except KeyboardInterrupt:
    pass
#model_admission.save_weights('adm_model')