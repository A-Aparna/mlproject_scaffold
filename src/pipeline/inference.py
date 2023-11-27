import pandas as pd
import json
import os
#os.chdir('C:\\Aparna\\mlops')
from src.helper import load_obj, save_object




class inference:
    def __init__(self):
        self.transform_path=os.path.join(os.getcwd(),'model_artifacts\\transformation.pkl')
        self.model_path = os.path.join(os.getcwd(),'model_artifacts\\model.pkl')
        self.transform = load_obj(self.transform_path)
        self.model = load_obj(self.model_path)

    def predict(self,data):
        f_data = pd.DataFrame(data)#format_data(data)
        t_data = self.transform.transform(f_data)
        prediction = self.model.predict(t_data)
        return prediction
    
class format_data:
    def __init__(self,data):
        self.data = data
        return self.data_to_df()
    
    def data_to_df(self):            
        return pd.DataFrame(self.data)