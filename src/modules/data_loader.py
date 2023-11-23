import os
##os.chdir('C:\\Aparna\\mlops')
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
print(os.listdir())
print(os.getcwd())
from src.logger import logging

STAGE_NAME="data_loader"
@dataclass
class data_loader_config:
    def __init__(self,data_type='csv'):
        self.data_type = data_type
    data_path: str = os.path.join('data_artifacts','raw_data.csv')
    train_data_path: str = os.path.join('data_artifacts','train_data.csv')
    test_data_path: str = os.path.join('data_artifacts','test_data.csv')


class data_loader:
    def __init__(self,data_path,data_type='csv'):
        self.data_type = data_type
        self.data_path = data_path
        self.data_config = data_loader_config(self.data_type)
        print('data_config',self.data_config)

    def load(self):
        if self.data_type=='csv':
            self.load_csv()

    def load_csv(self):
        logging.info(f"loading data from {self.data_path}")
        df=pd.read_csv(self.data_path)
        os.makedirs(os.path.dirname(self.data_config.data_path),exist_ok=True)
        df.to_csv(self.data_config.data_path,index=False,header=True)
        logging.info(f"data saved at {self.data_config.data_path}")

        train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
        train_set.to_csv(self.data_config.train_data_path,index=False,header=True)
        test_set.to_csv(self.data_config.test_data_path,index=False,header=True)


if __name__=='__main__':
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_loader_obj = data_loader('data/stud.csv')
    data_loader_obj.load()
    logging.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<")