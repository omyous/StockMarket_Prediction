from matplotlib import pyplot
from tensorflow_core import sqrt
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.losses import mean_squared_error
from tensorflow_core.python.keras.saving import load_model

from src.scrapping.attention import *
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src.scrapping.preprocessing import *
def create_model(input_timesteps=5,
    input_dim =6):
    # Use scikit-learn to grid search the batch size and epochs

    """ data = LSTM_data()
    X_train, y_train, X_test, y_test = data.get_memory()
    input_timesteps = data.lag
    input_dim = X_train.shape[2]"""
    drop_out = .2


    optim = Adam(lr=1e-3)
    i = Input(shape=(input_timesteps, input_dim))
    x =  LSTM(200,
                   activation='relu',
                   return_sequences=True,
                   input_shape=(input_timesteps, input_dim),

                   bias_regularizer=regularizers.l2(1e-4),
                   activity_regularizer=regularizers.l2(1e-5)
                   )(i)
    x = Dropout(.2)(x)
    x = LSTM(200,
             activation='relu',
             return_sequences=True,
             input_shape=(input_timesteps, input_dim),

             bias_regularizer=regularizers.l2(1e-4),
             activity_regularizer=regularizers.l2(1e-5)
             )(x)
    x = Dropout(.2)(x)
    x = LSTM(200,
             activation='relu',
             return_sequences=True,
             input_shape=(input_timesteps, input_dim),

             bias_regularizer=regularizers.l2(1e-4),
             activity_regularizer=regularizers.l2(1e-5)
             )(x)
    x = Dropout(0.2)(x)
    x = LSTM(50,
             activation='relu',
             return_sequences=True,
             input_shape=(input_timesteps, input_dim),

             bias_regularizer=regularizers.l2(1e-4),
             activity_regularizer=regularizers.l2(1e-5)
             )(x)
    x = Dropout(.2)(x)
    x = LSTM(50,
             activation='relu',
             return_sequences=True,
             input_shape=(input_timesteps, input_dim),

             bias_regularizer=regularizers.l2(1e-4),
             activity_regularizer=regularizers.l2(1e-5)
             )(x)
    x = attention_3d_block(x)
    x = Dropout(.2)(x)

    x = Dense(1, activation='relu')(x)
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='mse',
                  optimizer=optim,
                  metrics=["accuracy"])
    # LSTM MODEL

    return model

def train_model(X_train, y_train, X_test, y_test ):

    dense_output = 1
    epochs = 800
    batch_size = 128
    model = create_model(input_dim=X_train.shape[2])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=5, min_lr=0.00055)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr],
                        validation_data=(X_test, y_test))
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    #model.save('data/weights/lstm-senti-word2-attn.h5')

def evaluate(X_test, y_test, scaler):

    df = pd.read_csv("data/merged.csv")[["Close","High","Low","Open","Volume","positive","negative"]]
    model = load_model('data/weights/lstm-senti-word2.h5')

    yhat = model.predict(X_test)

    inv_yhat = scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = y_test.reshape((len(y_test), 1))
    inv_y = scaler.inverse_transform(test_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
if __name__=="__main__":
    data = LSTM_data()
    X_train, y_train, X_test, y_test = data.get_memory()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_model(X_train, y_train, X_test, y_test )


    """"""

    """model_ = KerasRegressor(build_fn=model, epochs=100, batch_size=10, verbose=0)

    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model_, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)"""