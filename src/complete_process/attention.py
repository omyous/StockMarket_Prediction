from tensorflow_core.python.keras.engine import Layer
import tensorflow.keras.backend as K


class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        #create the vector of weights of size equal to number of features
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")

        #create the vectors of biases
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        #compute the dot product between the weights vectors and the input sequence and then pass it to the tan hyperbolic function
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        #pass the tan result to a softmax function
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()