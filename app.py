import os
import pandas as pd
#os.chdir('C:\\Aparna\\mlops')
import uvicorn
from fastapi import FastAPI

from src.pipeline.inference import inference
from src.helper import load_training_data
from post_data_validation import DataValidation
app = FastAPI()

@app.get('/')
def index():
    print('index page')
    return {'message': 'Let''s predict the score'}

@app.get('/train')
def train():
    print("loading training data from drive")
    load_training_data()
    print('training started')
    os.system('dvc repro')
    return {'message': 'Training completed'}

@app.post('/predict')
def predict(data:DataValidation):
    print(data)
    data=data.dict()
    for key,value in data.items():
        data[key]=[value]
    inf = inference()
    result = inf.predict(data)
    return {'score is':result[0]}

    
def predict_cli():
    data={}
    data['gender']=[input("what is the gender-male/female?")]
    data['race_ethnicity']=[input("what is the race- group A,B,C,D,E?")]
    data['parental_level_of_education']=[input("what is the parental level of education?-master's degree,bachelor's degree,associate's degree,some college,some high school,high school")]
    data['lunch']=[input("what is the type of lunch?-standard,free/reduced")]
    data['test_preparation_course']=[input("what is the test preparation course?-none,completed")]
    data['reading_score']=[input("what is the reading score out of 100?")]
    data['writing_score']=[input("what is the writing score out of 100?")]
    inf = inference()
    result = inf.predict(data)
    return result
    
if __name__=="__main__":
    #result = predict_cli()
    uvicorn.run(app,host='127.0.0.1',port=8080)