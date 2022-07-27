import seaborn as sns
import scipy
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

seed = np.random.randint(1, 1234)
tf.random.set_seed(seed=seed)
print(seed)


def mean_absolute_percentage_error_01(Y_test, Y_testpredict):
    Y_test, Y_testpredict = np.array(Y_test), np.array(Y_testpredict)
    return np.mean(np.abs((Y_test - Y_testpredict) / Y_test)) * 100


# Carregamento dos dados de treino e teste
dataset_01 = pd.read_excel(
    "/content/drive/MyDrive/arquivos csv/serra_do_facao_treino.xlsx", parse_dates=['Data'], index_col='Data')
dataset_02 = pd.read_excel(
    "/content/drive/MyDrive/arquivos csv/serra_do_facao_teste.xlsx", parse_dates=['Data'], index_col='Data')

# Normalização dos dados de treino e teste
scaler = MinMaxScaler(feature_range=(0.1, 0.9))
X_train = dataset_01.drop(
    columns=['12D', '11D', '10D', '9D', '8D', '7D', '6D', '5D', '4D', '3D', '2D', '1D'])
X_train = scaler.fit_transform(X_train)

Y_train = dataset_01[['12D', '11D', '10D', '9D',
                      '8D', '7D', '6D', '5D', '4D', '3D', '2D', '1D']]
Y_train = scaler.fit_transform(Y_train)

X_test = dataset_02.drop(
    columns=['12D', '11D', '10D', '9D', '8D', '7D', '6D', '5D', '4D', '3D', '2D', '1D'])
X_test = scaler.fit_transform(X_test)

Y_test = dataset_02[['12D', '11D', '10D', '9D',
                     '8D', '7D', '6D', '5D', '4D', '3D', '2D', '1D']]
Y_test = scaler.fit_transform(Y_test)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


def create_dataset(X, Y, time_steps=1):
    Xs, Ys = [], []

    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        Ys.append(Y[i+time_steps])

    return np.array(Xs), np.array(Ys)


X_test, Y_test = create_dataset(X_test, Y_test, 60)
X_train, Y_train = create_dataset(X_train, Y_train, 60)
print('X_train.shape: ', X_train.shape)
print('Y_train.shape: ', Y_train.shape)
print('X_test.shape: ', X_test.shape)
print('X_test.shape: ', Y_test.shape)

# Create BiLSTM model


def create_model_bilstm(units1, units2):
    model = Sequential()
    # First layer of BiLSTM
    model.add(Bidirectional(LSTM(units=units1, return_sequences=True),
              input_shape=(X_train.shape[1], X_train.shape[2])))
    # Second layer of BiLSTM
    model.add(Bidirectional(LSTM(units=units2)))
    model.add(Dense(units=12))
    # Compile model
    #model.compile(loss='mse', optimizer='adam')
    model.compile(loss='huber_loss',
                  optimizer=RMSprop(),
                  metrics=['mean_absolute_percentage_error'])
    return model

# Create LSTM or GRU model


def create_model(units1, units2, model_type):
    model = Sequential()
    # First layer of LSTM
    model.add(model_type(units=units1, return_sequences=True,
              input_shape=[X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.1))
    # Second layer of LSTM
    model.add(model_type(units=units2))
    model.add(Dropout(0.1))
    model.add(Dense(units=12))
    # Compile model
    #model.compile(loss='mse', optimizer='adam')
    model.compile(loss='huber_loss',
                  optimizer=RMSprop(),
                  metrics=['mean_absolute_percentage_error'])
    return model


model_bilstm = create_model_bilstm(256, 128)
model_lstm = create_model(256, 128, LSTM)
model_gru = create_model(256, 128, GRU)


def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=10)
    history = model.fit(X_train, Y_train, epochs=25,
                        validation_split=0.2, batch_size=128,
                        shuffle=False, callbacks=[early_stop])
    return history


history_bilstm = fit_model(model_bilstm)
history_lstm = fit_model(model_lstm)
history_gru = fit_model(model_gru)


def score(model, model_name):
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("\nTest score " + model_name + ":", score[0])
    print('Test mean_absolute_percentage_error ' + model_name + ":", score[1])


score(model_bilstm, 'BiLSTM')
score(model_lstm, 'LSTM')
score(model_gru, 'GRU')


def plot_mean_absolute_percentage_error(history, model_name):
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('model mean_absolute_percentage_error for ' + model_name)
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')


def plot_loss(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper left')

# plot_mean_absolute_percentage_error(history_bilstm, 'BiLSTM')
# plot_loss (history_bilstm, 'BiLSTM')

# plot_mean_absolute_percentage_error(history_lstm, 'LSTM')
# plot_loss(history_lstm, 'LSTM')


plot_mean_absolute_percentage_error(history_gru, 'GRU')
plot_loss(history_gru, 'GRU')

Y_test = scaler.inverse_transform(Y_test)
Y_train = scaler.inverse_transform(Y_train)

# Y_trainpredict = model_bilstm.predict(X_train)
# Y_testpredict = model_bilstm.predict(X_test)

# Y_trainpredict = model_lstm.predict(X_train)
# Y_testpredict = model_lstm.predict(X_test)

Y_trainpredict = model_gru.predict(X_train)
Y_testpredict = model_gru.predict(X_test)

Y_trainpredict = scaler.inverse_transform(Y_trainpredict)
Y_testpredict = scaler.inverse_transform(Y_testpredict)

# Test Score
testeScore_01 = mean_absolute_percentage_error_01(
    Y_test[:, 3:10], Y_testpredict[:, 3:10])
print(testeScore_01)

MED_Y_test = np.average(Y_test[:, 3:10], axis=1)
# print(MED_Y_test)
MED_Y_pred = np.average(Y_testpredict[:, 3:10], axis=1)
# print(MED_Y_test)

result = np.argmax(MED_Y_test[0:240])
result1 = np.argmax(MED_Y_pred[0:240])
print(result, result1)

result2 = np.argmax(MED_Y_test[240:480])
result3 = np.argmax(MED_Y_pred[240:480])
print(result2, result3)

result4 = np.argmax(MED_Y_test[480:720])
result5 = np.argmax(MED_Y_pred[480:720])
print(result4, result5)

plt.plot(range(240), MED_Y_test[0:240])
plt.plot(range(240), MED_Y_pred[0:240])

plt.title('Ocorrido x Previsto')
plt.ylabel('vazão')
plt.xlabel('time')
plt.legend(['test', 'predict'], loc='upper left')
plt.show()

plt.plot(range(240), MED_Y_test[240:480])
plt.plot(range(240), MED_Y_pred[240:480])
plt.title('Ocorrido x Previsto')
plt.ylabel('vazão')
plt.xlabel('time')
plt.legend(['test', 'predict'], loc='upper left')
plt.show()

plt.plot(range(180), MED_Y_test[480:660])
plt.plot(range(180), MED_Y_pred[480:660])
plt.title('Ocorrido x Previsto')
plt.ylabel('vazão')
plt.xlabel('time')
plt.legend(['test', 'predict'], loc='upper left')
plt.show()
