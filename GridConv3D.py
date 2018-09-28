
from keras.layers import Conv3D,Input,MaxPooling3D,Dropout,Flatten,Dense
from keras.models import Sequential
from keras.losses import MSE
from keras.optimizers import adam

import keras.backend as K


class GridConv3D():

    def __init__(self,
                 sizeX,
                 sizeY,
                 sizeZ,
                 n_channel
                 ):
        '''

        x,y,z是编码的Grid的格子数目
        n_channel是atom case的数目，使用通道表示
        '''
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.sizeZ = sizeZ
        self.n_channel = n_channel
        self.model = None

    def fit(self,X,y,epochs,batch_size):
        if self.model is None:
            self.build()
        self.model.fit(X,y,epochs=epochs,batch_size=batch_size)

    def predict(self,X):
        return self.model.predict(X)

    def build(self):
        # 先保证过拟合，所以不加dropout
        model = Sequential()

        model.add(Conv3D(
                       input_shape=(self.sizeX,self.sizeY,self.sizeZ,self.n_channel),
                       dtype=K.floatx(),
                       filters=64,
                       kernel_size=8,
                       strides=(1,1,1),
                       padding="SAME",
                       activation="relu",
                       data_format="channels_last",
                       kernel_initializer="glorot_uniform",
                       bias_initializer="zeros"

                       ))
        model.add(MaxPooling3D(
            pool_size=2,
            strides=(2, 2, 2),
            padding="SAME"
        ))
        #model.add(Dropout(0.25))
        model.add(Conv3D(filters=64,
                       kernel_size=4,
                       strides=(1,1,1),
                       padding="SAME",
                       activation="relu",
                       data_format="channels_last",
                       kernel_initializer="glorot_uniform",
                       bias_initializer="zeros"
                       ))
        model.add(MaxPooling3D(
            pool_size=2,
            strides=(2, 2, 2),
            padding="SAME"
        ))
        #model.add(Dropout(0.25))
        model.add(Conv3D(filters=128,
                       kernel_size=2,
                       strides=(1, 1, 1),
                       padding="valid",
                       activation="relu",
                       data_format="channels_last",
                       kernel_initializer="glorot_uniform",
                       bias_initializer="zeros"
                       ))
        #model.add(Dropout(0.25))
        model.add(MaxPooling3D(
            pool_size=2,
            strides=(2, 2, 2),
            padding="SAME"
        ))
        model.add(Conv3D(filters=256,
                         kernel_size=2,
                         strides=(1, 1, 1),
                         padding="valid",
                         activation="relu",
                         data_format="channels_last",
                         kernel_initializer="glorot_uniform",
                         bias_initializer="zeros"
                         ))
        model.add(MaxPooling3D(
            pool_size=2,
            strides=(2, 2, 2),
            padding="SAME"
        ))
        model.add(Flatten())
        model.add(Dense(256,activation="relu"))
        #model.add(Dropout(0.2))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dropout(0.2))
        #model.add(Dense(64,activation="relu"))
        #model.add(Dropout(0.2))
        #model.add(Dense(32, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(16,activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(1))
        model.summary()

        model.compile(loss=MSE,optimizer="adam",metrics=["mse"])

        self.model = model







if __name__ == '__main__':

    a = GridConv3D(60,60,60,3)
    a.build()