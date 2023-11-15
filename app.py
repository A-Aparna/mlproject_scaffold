import pickle 
from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
os.chdir('C:\\Aparna\\mlops')

from src.pipeline.inference import inference

app = Flask(__name__)

@app.route('/')
def render():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=="GET":
        return render_template('home.html')
    
if __name__=="__main__":
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
    print(result)