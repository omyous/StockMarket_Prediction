from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow_core.python.keras import Sequential

from src.scrapping.attention import *
from src.scrapping.preprocessing import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError as MAPE
from tensorflow.keras.metrics import Accuracy

ACTIVATION = "tanh"
NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-6
BATCH_SIZE =64
BIAIS_REG = 3e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0005)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)

def plot_train_loss(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('train loss vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

def evaluate(regressor, X_test, Y_test, dataset_object, name:str, senti:str):
    yhat = regressor.predict(X_test)
    Y_test_ = Y_test[:,0]
    yhat_= yhat[:,0]
    print('Test RMSE: %.3f' % mean_squared_error(Y_test_, yhat_).numpy())
    print('Test MAE: %.3f' % mean_absolute_error(Y_test_, yhat_).numpy())
    #print(MAPE(Y_test, yhat))
    # invert scaling for forecast
    inv_yhat = dataset_object.y_scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = Y_test.reshape((len(Y_test), 1))
    inv_y = dataset_object.y_scaler.inverse_transform(test_y)
    inv_y = inv_y[:, 0]
    print(inv_y.shape, inv_yhat.shape, dataset_object.test_dates.shape)
    pd.DataFrame({"predictions": inv_yhat,
                  "Close": inv_y,
                  "Date": dataset_object.test_dates
                  }).to_csv("data/"+name+"_senti_"+senti+".csv", index=False)
    # calculate RMSE
    rmse = mean_squared_error(inv_y, inv_yhat)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mean_absolute_error(inv_y, inv_yhat))

    plot_preds(inv_yhat, inv_y)

def plot_preds(inv_yhat, inv_y):
    pyplot.plot(inv_yhat)
    pyplot.plot(inv_y)
    pyplot.title('y_preds vs real y')
    pyplot.ylabel('preds')
    pyplot.xlabel('epoch')
    pyplot.legend(['predictions', 'real values'], loc='upper right')
    pyplot.show()
def free_attn_lstm(dataset_object: LSTM_data):
    X_train, X_test, Y_train, Y_test = dataset_object.get_memory()
    X_train, X_test = X_train[:, :, :-12], X_test[:, :, :-12]
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=NEURONS,
                       return_sequences=True,
                       activation=ACTIVATION,
                       recurrent_activation="sigmoid",
                       input_shape=(X_train.shape[1], X_train.shape[2]),
                       bias_regularizer=regularizers.l2(BIAIS_REG),
                       activity_regularizer=regularizers.l2(L2)
                       ))
    regressor.add(Dropout(DROPOUT))
    regressor.add(LSTM(units=NEURONS,
                       activation=ACTIVATION,
                       recurrent_activation="sigmoid",
                       return_sequences=True,
                       bias_regularizer=regularizers.l2(BIAIS_REG),
                       activity_regularizer=regularizers.l2(L2)

                       ))
    regressor.add(Dropout(DROPOUT))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=NEURONS,
                       activation=ACTIVATION,
                       recurrent_activation="sigmoid",
                       bias_regularizer=regularizers.l2(BIAIS_REG),
                       activity_regularizer=regularizers.l2(L2)
                  ))
    regressor.add(Dropout(DROPOUT))
    # Adding the output layer
    regressor.add(Dense(units=1,
                        activation='relu',
                        bias_regularizer=regularizers.l2(BIAIS_REG),
                       activity_regularizer=regularizers.l2(L2)
                        )
                  )
    optim = Adam()
    # Compiling the RNN
    regressor.compile(optimizer=optim, loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history= regressor.fit(X_train,
                           Y_train,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data=(X_test, Y_test),
                           callbacks=[REDUCE_LR, EARLY_STOP]
                           )
    regressor.save("data/weights/free_attn_lstm_no_senti")
    plot_train_loss(history)
    evaluate(regressor,X_test,Y_test, dataset_object,name="free_attn_lstm", senti="no")


# ---------------------------------------------- Attention based lstm ----------------------------------------------#
def attn_many_to_one(dataset_object: LSTM_data):

    X_train, X_test, Y_train, Y_test = dataset_object.get_memory()
    X_train, X_test = X_train[:, :, :-10], X_test[:, :, :-10]

    i = Input(shape=(X_train.shape[1], X_train.shape[2]))

    att_in = LSTM(NEURONS,
                    return_sequences=True,
                    activation=ACTIVATION,
                  recurrent_activation="sigmoid",
                    activity_regularizer=regularizers.l2(L2),
                    bias_regularizer=regularizers.l2(BIAIS_REG),
                  )(i)

    att_in = LSTM(NEURONS,
                  return_sequences=True,
                  activation=ACTIVATION,
                  recurrent_activation="sigmoid",
                  activity_regularizer=regularizers.l2(L2),
                  bias_regularizer=regularizers.l2(BIAIS_REG),
                  )(att_in)
    att_in = LSTM(NEURONS,
                  return_sequences=True,
                  activation=ACTIVATION,
                  recurrent_activation="sigmoid",
                  activity_regularizer=regularizers.l2(L2),
                  bias_regularizer=regularizers.l2(BIAIS_REG),
                  )(att_in)
    att_out = attention()(att_in)
    att_out = Dropout(DROPOUT)(att_out)
    outputs = Dense(1,
                    activation='relu',
                    trainable=True,
                    bias_regularizer=regularizers.l2(BIAIS_REG),
                    activity_regularizer=regularizers.l2(L2)
                    )(att_out)

    model = Model(inputs=[i], outputs=[outputs])
    optim = Adam()
    model.compile(optimizer=optim,
                  loss=['mean_squared_error']
                  )

    # Fitting the RNN to the Training set
    history = model.fit(X_train, Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, Y_test),
                        callbacks=[EARLY_STOP, REDUCE_LR]
                        )
    model.save("data/weights/attn_based_lstm_no_senti")
    plot_train_loss(history)
    evaluate(model,X_test,Y_test, dataset_object,name="attn_evaluate", senti="no")

#------------------------------------------- Dense net ----------------------------------#
def dense_net(dataset_object:LSTM_data):
    X_train, X_test, Y_train, Y_test = dataset_object.get_splited_data()
    #X_train, X_test = X_train[:, :-12], X_test[:, :-12]
    regressor = Sequential()

    regressor.add(Dense(units=EPOCHS,
                        activation='relu',
                        bias_regularizer=regularizers.l2(BIAIS_REG),
                        activity_regularizer=regularizers.l2(L2)
                        ))

    regressor.add(Dense(units=EPOCHS,
                        activation='relu',
                        bias_regularizer=regularizers.l2(BIAIS_REG),
                        activity_regularizer=regularizers.l2(L2)
                        ))
    regressor.add(Dense(units=EPOCHS,
                        activation='relu',
                        bias_regularizer=regularizers.l2(BIAIS_REG),
                        activity_regularizer=regularizers.l2(L2)
                        ))
    regressor.add(Dropout(DROPOUT))
    regressor.add(Dense(units=1,
                        activation='relu',
                        bias_regularizer=regularizers.l2(BIAIS_REG),
                        activity_regularizer=regularizers.l2(L2)
                        ))
    optim = Adam()
    # Compiling the RNN
    regressor.compile(optimizer=optim, loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history= regressor.fit(X_train,
                           Y_train,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data=(X_test, Y_test),
                           callbacks=[EARLY_STOP, REDUCE_LR])
    regressor.save("data/weights/dense_senti")
    plot_train_loss(history)
    evaluate(regressor, X_test,Y_test, dataset_object,name="dense", senti="yes")

if __name__=="__main__":

    #attn_many_to_one(LSTM_data())
    #free_attn_lstm(LSTM_data())
    dense_net(LSTM_data())

"""data = pd.read_csv('data/merged.csv',
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

    print(X_train_reshaped.shape, X_test_reshaped.shape)"""
"""
    sauvegarde1:
    NEURONS: int = 300
DROPOUT:int = .2
EPOCHS :int = 800
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(patience=10)



-------------------------2
NEURONS: int = 300
DROPOUT:int = .2
EPOCHS :int = 800
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(patience=10)

Test RMSE: 7.545
------------------------3
NEURONS: int = 300
DROPOUT:int = .2
EPOCHS :int = 800
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=10,
                        restore_best_weights=True)
                        
Rmse : 7.328
    ################################L2 activity = 1e-5 pour les 3 précédents 
    
    
    
    from src.scrapping.ML_Models import *
from tensorflow_core.python.keras.layers import Dropout
from src.scrapping.attention import *
from src.scrapping.preprocessing import *
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping

NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-3
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=10,
                        restore_best_weights=True)

Test RMSE: 6.110


sauvegarde 4
2 lstm
NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-6

REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.00015)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        #mode="min",
                        patience=5,
                        restore_best_weights=True)
RMSE: 6.677, overfiting réduit        


# version 5
NEURONS: int = 600
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-6
BATCH_SIZE =64
BIAIS_REG = 3e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)      
                        
Test RMSE: 5.082
                        
                        
version 6

NEURONS: int = 600
DROPOUT:int = .3
EPOCHS :int = 800
L2 = 1e-6#1e-6
BATCH_SIZE =128
BIAIS_REG = 3e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(#monitor='val_loss',
                        #mode="min",
                        patience=5,
                        restore_best_weights=True)
                        
Test RMSE: 6.444


#####################################
5 lstm
pas d'overfiting
rmse = 11.701
NEURONS: int = 600
DROPOUT:int = .3
EPOCHS :int = 800
L2 = 1e-6
BATCH_SIZE =64
BIAIS_REG = 3e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0001)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)


###########################"
3 lstm
RMSE : 6.644
NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-5
BATCH_SIZE =64
BIAIS_REG = 3e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0005)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)


##############################################################################"
3lstm, RMSE = 6.1
NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-5
BATCH_SIZE =64
BIAIS_REG = 6e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0005)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)
                        
                        
                        
                        
                        
                        
#####################################################"

RMSE: 5.02
3LSTM
ACTIVATION = "tanh"
NEURONS: int = 500
DROPOUT:int = .2
EPOCHS :int = 800
L2 = 1e-5
BATCH_SIZE =64
BIAIS_REG = 6e-3#1e-4
REDUCE_LR = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              mode="min",
                              min_lr=.0005)#0.00055
EARLY_STOP= EarlyStopping(monitor='val_loss',
                        mode="min",
                        patience=5,
                        restore_best_weights=True)

"""