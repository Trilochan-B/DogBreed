import sys
import numpy
import cv2
import numpy as np

from src.exception import CustonmException
from src.utill import load_object


class prediction:
    def __init__(self):
        self.moselPath = "artifacts/BreadModel.pkl"
        self.labelPath = "artifacts/BreedName.pkl"

    def predict(self,data):
        try:
            image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            image_r = cv2.resize(image, (28,28))/255
            img_r = image_r[np.newaxis,...]
            model = load_object(self.path)
            breed = load_object()
            pred = model.predict(img_r)
            name = breed[np.argmax(pred)]
            return name
        except Exception as e:
            raise CustonmException(e,sys)
