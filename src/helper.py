import os
#os.chdir('C:\\Aparna\\mlops')
import pandas as pd
import pickle
import gdown

def hello():
    return "hello"

def get_columns_type(df,type):
    return df.select_dtypes(include=type).columns.tolist()

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print('saved object at ',file_path )
    except Exception as e:
        raise e
    
def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        return obj
    except Exception as e:
        raise e


def load_training_data():
    download_dir='./data/stud.xlsx'
    
    file_id='https://docs.google.com/spreadsheets/d/1aqcCFVY-5PWNUGU51JVmcn1QsGB3YOlDzAotc_3wsRk/edit?usp=drive_link'.split('/')[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    os.makedirs("data", exist_ok=True)
    gdown.download(prefix+file_id,download_dir)
    pd.read_excel('data/stud.xlsx').to_csv('data/stud.csv', index=False)
