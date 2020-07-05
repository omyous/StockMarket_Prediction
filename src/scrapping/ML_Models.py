from tensorflow.keras import models

def attn_based_lst(X_train, X_test, Y_train, Y_test):
    print("this a anttention base model")
    model = models.load_model("data/weights/attn_based_lstm")
    return
def light_lstm(X_train, X_test, Y_train, Y_test):
    print("this a light lstm model")
    return
def denseNet(X_train, X_test, Y_train, Y_test):
    print("this a dense net model")
    return