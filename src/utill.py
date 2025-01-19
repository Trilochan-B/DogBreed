import os
import sys
from src.exception import CustonmException
from src.logger import logging
import pickle
from sklearn.model_selection import train_test_split

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path,"wb") as file :
            pickle.dump(obj, file)
            logging.info("Object saved successfully")
    except Exception as e:
        raise CustonmException(e,sys)

    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustonmException(e,sys)
    
def dataSplit(x,y):
    try:
       x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
       return  x_train,x_test,y_train,y_test
    except Exception as e:
        raise CustonmException(e, sys)