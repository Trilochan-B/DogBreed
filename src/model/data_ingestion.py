import os
import cv2
import sys
from src.logger import logging
from src.exception import CustonmException

from src.model.data_tranform import dataTransform
from src.model.model import modelTrain
from src.utill import dataSplit


class dataIngestion ():
    def __init__(self):
        pass
    def loadData(self,path):
        try:
            dataset = []
            label = []

            logging.info("Data ingestion started")

            for breed in os.listdir(path):
                for image in os.listdir(os.path.join(path,breed)):
                    img = cv2.imread(os.path.join(path,breed,image))
                    dataset.append(cv2.resize(img, (28,28)))
                    label.append(breed)

            logging.info("Data ingestion completed")

            return dataset,label
        except Exception as e:
            raise CustonmException(e,sys)
        
if __name__ == "__main__":
    dataLoad = dataIngestion()
    dataset, labels = dataLoad("") # please provide data path here
    transobj = dataTransform()
    x, y = transobj.transform(dataset, labels)
    x_train, x_test, y_train, y_test = dataSplit(x,y)
    trainObj = modelTrain()
    model = trainObj.tarin(x_train, y_train)
    trainObj.evaluateModel(model,x_test, y_test)


