from src.exception import CustonmException
from src.logger import logging
from src.utill import save_object

from tensorflow import keras
from dataclasses import dataclass
import os
import sys

@dataclass
class modelConfig:
    modelPath: str = os.path.join("artifacts","BreadModel.pkl")

class modelTrain:
    def __init__(self):
        self.model_path = modelConfig()

    def tarin(self,x_train,y_train):
        try:
            logging.info("model training initiated")
            cnn = keras.Sequential([ 
                 #cnn 
                 keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,3)), keras.layers.MaxPooling2D((2,2)), 
                 keras.layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu'), 
                 keras.layers.MaxPooling2D((2,2)),

                #dence
                keras.layers.Flatten(input_shape=(28,28,3)),
                keras.layers.Dense(1000,activation='relu'),
                keras.layers.Dense(100,activation='relu'),
                keras.layers.Dense(10,activation='sigmoid')
            ])

            cnn.compile( optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'] )
            cnn.fit(x_train,y_train, epochs=100)

            logging.info("model training completed")

            return cnn
        except Exception as e:
            raise CustonmException(e,sys)
        
    def evaluateModel(self, model,x,y):
        try:
            result = model.evaluate(x,y)
            if result[1] >= 0.85:
                save_object(model, self.model_path.modelPath)
                logging.info(f"Model was built with accuracy = {result[1]}")
            else:
                logging.info("Accuracy of the model is low please modeify the structure")
        except Exception as e:
            raise CustonmException(e,sys)