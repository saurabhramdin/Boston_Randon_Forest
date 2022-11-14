import config
import json
import pickle
import numpy as np

class BostonHousePrice():
    def __init__(self,LSTAT,RM,DIS,CRIM,PTRATIO,AGE,B,NOX):
        self.LSTAT = LSTAT
        self.RM = RM
        self.DIS = DIS
        self.CRIM = CRIM
        self.PTRATIO = PTRATIO
        self.AGE = AGE
        self.B = B
        self.NOX = NOX

    def load_model(self):
        with open(config.ENCODER_FILE_PATH,'r') as f:
            self.encoder = json.load(f)
        with open(config.MODEL_FILE_PATH,'rb') as f:
            self.model = pickle.load(f)
        with open(config.SCALER_FILE_PATH,'rb') as f:
            self.scaler = pickle.load(f)

    def predict_price(self):
        self.load_model()
        test_arr = np.zeros(len(self.encoder['columns']))
        test_arr[0] = self.LSTAT
        test_arr[1] = self.RM
        test_arr[2] = self.DIS
        test_arr[3] = self.CRIM
        test_arr[4] = self.PTRATIO
        test_arr[5] = self.AGE
        test_arr[6] = self.B
        test_arr[7] = self.NOX

        test_arr = self.scaler.transform([test_arr])
        price = self.model.predict(test_arr)[0]

        return price