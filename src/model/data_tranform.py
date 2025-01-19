import numpy as np
from tensorflow import keras
from src.utill import save_object
from dataclasses import dataclass
import os
import sys

from src.logger import logging
from src.exception import CustonmException

@dataclass
class dataConfig:
    labelPath : str = os.path.join('artifacts','BreedName.pkl')



class dataTransform:
    def __init__(self):
        self.data_config = dataConfig()

    def transform(self,dataset,label):
        try:
            logging.info("Data transformation initiated")
            dataset = np.array(dataset)
            classes = np.unique(label)
            x_scale = dataset/255
            for i,name in enumerate(label) :
                label[i] = np.where(classes == name)[0][0]
                
            y_cat = keras.utils.to_categorical(label, num_classes=10)
            save_object(classes, self.data_config.labelPath)
            logging.info("Data transformation completed")
            return x_scale, y_cat
        except Exception as e:
            raise CustonmException(e,sys)