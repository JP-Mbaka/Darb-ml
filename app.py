"""
 @author: Mbaka JohnPaul

 """

import uvicorn
from fastapi import FastAPI
from Darb import Darb
import numpy as np
import pickle 
import pandas as pd

app = FastAPI()
pickle_in = open("random_model.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, welcome to Darb-API'}

@app.post('/predict')
def predict_diabetes(data:Darb):
    data = data.dict()
    preg = data['preg']
    plas = data['plas']
    pres = data['pres']
    skin = data['skin']
    insu = data['insu']
    mass = data['mass']
    pedi = data['pedi']
    age = data['age']

    prediction = classifier.predict([[preg,plas,pres,skin,insu,mass,pedi,age]])
    return{
        'prediction': str(prediction)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port= 8000)
else:
    uvicorn.run(app, host="0.0.0.0", port=8000)
