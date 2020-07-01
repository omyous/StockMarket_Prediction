import time
from src.scrapping.ML_Models import *
from tensorflow_core.python.keras.layers import Dropout
from src.scrapping.attention import *

from src.scrapping.preprocessing import *
def many_to_one():
    data = pd.read_csv('data/merged.csv',
                                engine='python',
                                index_col="Date"
                                )
    Y = data[["Close"]]
    Y.columns = ["Y"]
    #shift the features

    cols = ["High","Low","Open","Volume","Adj Close","positive","negative"]
    dres = data
    lag_value = 5

    # shift to get all the data from X
    for i in range(1, lag_value + 1):
        dtemp = data[cols].shift(periods=i)
        dtemp.columns = [c + '_lag' + str(i) for c in cols]
        dres = dres.merge(dtemp, on='Date')
    X = dres

    # shift the dependant variable
    cols = Y.columns
    dy = pd.DataFrame(index=Y.index)
    for i in range(1, lag_value+1):
        dytemps = Y.shift(periods=i)
        dytemps.columns = [c + '_lag' + str(i) for c in cols]
        dy = dy.merge(dytemps, on='Date')
    X = X.merge(dy, on='Date')
    X = X.dropna()

    Y =X[["Close"]].values
    X = X.drop(["Close"], axis=1).values

    X_train, X_test = X[:int(X.shape[0] * .8), :], X[int(X.shape[0] * .8):, :]

    y_train, y_test = Y[:int(X.shape[0] * .8), :], Y[int(X.shape[0] * .8):, :]

    scalerX = MinMaxScaler()
    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)

    scalerY = MinMaxScaler()
    Y_train_scaled = scalerY.fit_transform(y_train)
    Y_test_scaled = scalerY.transform(y_test)

    X_train_reshaped = np.zeros(shape=(int(X_train.shape[0]/1), 1, X_train.shape[1]), dtype=np.float32)
    X_test_reshaped = np.zeros(shape=(int(X_test.shape[0]/1), 1, X_train.shape[1]), dtype=np.float32)

    for i in range(X_train_reshaped.shape[0]):
        X_train_reshaped[i] = X_train_scaled[i:i + 1, :]

    for i in range(X_test_reshaped.shape[0]):
        X_test_reshaped[i] = X_test_scaled[i:i + 1, :]

    print(X_train_reshaped.shape, X_test_reshaped.shape)
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=150,
                       return_sequences=True,
                       input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
                       bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5)
                       ))
    regressor.add(Dropout(0.22))
    regressor.add(LSTM(units=150,
                       return_sequences=True,
                       input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
                       bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5)

                       ))
    regressor.add(Dropout(0.22))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=150,
                       return_sequences=True,
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                  ))
    regressor.add(Dropout(0.22))


    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=150,
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5)
                       ))
    regressor.add(Dropout(0.22))


    # Adding the output layer
    regressor.add(Dense(units=1,
                        bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5)))
    optim = Adam()
    # Compiling the RNN
    regressor.compile(optimizer=optim, loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history= regressor.fit(X_train_reshaped, Y_train_scaled, epochs=800, validation_data=(X_test_reshaped, Y_test_scaled))
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    yhat = regressor.predict(X_test_reshaped)

    test_X = X_test_reshaped.reshape((X_test_reshaped.shape[0], X_test_reshaped.shape[2]))
    # invert scaling for forecast
    inv_yhat = scalerY.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    print(list(inv_yhat), "\n", list(Y_test_scaled))
    # invert scaling for actual
    test_y = Y_test_scaled.reshape((len(Y_test_scaled), 1))
    inv_y = scalerY.inverse_transform(test_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(inv_yhat)
    pyplot.plot(inv_y)
    pyplot.title('y_preds vs real y')
    pyplot.ylabel('preds')
    pyplot.xlabel('epoch')
    pyplot.legend(['predictions', 'real values'], loc='upper right')
    pyplot.show()
def attn_many_to_one():
    data = pd.read_csv('data/merged.csv',
                       engine='python',
                       index_col="Date"
                       )
    Y = data[["Close"]]
    Y.columns = ["Y"]
    # shift the features

    cols = ["High", "Low", "Open", "Volume", "Adj Close", "positive", "negative"]
    dres = data
    lag_value = 5

    # shift to get all the data from X
    for i in range(1, lag_value + 1):
        dtemp = data[cols].shift(periods=i)
        dtemp.columns = [c + '_lag' + str(i) for c in cols]
        dres = dres.merge(dtemp, on='Date')
    X = dres

    # shift the dependant variable
    cols = Y.columns
    dy = pd.DataFrame(index=Y.index)
    for i in range(1, lag_value + 1):
        dytemps = Y.shift(periods=i)
        dytemps.columns = [c + '_lag' + str(i) for c in cols]
        dy = dy.merge(dytemps, on='Date')
    X = X.merge(dy, on='Date')
    X = X.dropna()

    mu, sigma = 0, 0.1
    # creating a noise with the same dimension as the dataset (2,2)
    noise = np.random.normal(mu, sigma, X.shape)
    X = pd.DataFrame(np.add(X.values, noise), columns=X.columns)


    Y = X[["Close"]].values
    X = X.drop(["Close"], axis=1).values

    X_train, X_test = X[:int(X.shape[0] * .8), :], X[int(X.shape[0] * .8):, :]

    y_train, y_test = Y[:int(X.shape[0] * .8), :], Y[int(X.shape[0] * .8):, :]

    scalerX = MinMaxScaler()
    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)

    scalerY = MinMaxScaler()
    Y_train_scaled = scalerY.fit_transform(y_train)
    Y_test_scaled = scalerY.transform(y_test)

    X_train_reshaped = np.zeros(shape=(int(X_train.shape[0] / 1), 1, X_train.shape[1]), dtype=np.float32)
    X_test_reshaped = np.zeros(shape=(int(X_test.shape[0] / 1), 1, X_train.shape[1]), dtype=np.float32)

    for i in range(X_train_reshaped.shape[0]):
        X_train_reshaped[i] = X_train_scaled[i:i + 1, :]

    for i in range(X_test_reshaped.shape[0]):
        X_test_reshaped[i] = X_test_scaled[i:i + 1, :]

    print(X_train_reshaped.shape, X_test_reshaped.shape)
    i = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

    att_in = LSTM(300, return_sequences=True,  bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5) )(i)
    att_out = attention()(att_in)
    att_out = Dropout(0.2)(att_out)
    outputs = Dense(1, activation='relu', trainable=True,  bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5))(att_out)


    model = Model(inputs=[i], outputs=[outputs])
    optim = Adam()
    model.compile(optimizer=optim, loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history = model.fit(X_train_reshaped, Y_train_scaled, epochs=800,
                            validation_data=(X_test_reshaped, Y_test_scaled))
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    yhat = model.predict(X_test_reshaped)
    test_X = X_test_reshaped.reshape((X_test_reshaped.shape[0], X_test_reshaped.shape[2]))
    # invert scaling for forecast
    inv_yhat = scalerY.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = Y_test_scaled.reshape((len(Y_test_scaled), 1))
    inv_y = scalerY.inverse_transform(test_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(inv_yhat)
    pyplot.plot(inv_y)
    pyplot.title('y_preds vs real y')
    pyplot.ylabel('preds')
    pyplot.xlabel('epoch')
    pyplot.legend(['predictions', 'real values'], loc='upper right')
    pyplot.show()

if __name__=="__main__":

    attn_many_to_one()