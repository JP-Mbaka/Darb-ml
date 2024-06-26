"""
 @author: Mbaka JohnPaul

 """

import uvicorn
from fastapi import FastAPI
from Darb import Darb
import numpy as np
import pickle
# starteted modifying
import pandas as pd
from sklearn.model_selection import train_test_split

app = FastAPI()
# # model = joblib.load("random_mod.joblib")
# model_file_path = "random_model-1.pkl"
# # pickle_in = open("random_model-1.pkl","rb")
# # classifier = pickle.load(pickle_in)
# try:
#     with open(model_file_path, "rb") as pickle_file:
#         classifier = pickle.load(pickle_file,dtype='float64')
# except Exception as e:
#     print("Error loading pickled object:", e)

# Started modifying
df = pd.read_csv('diabete.csv')
X = df.drop('class',axis = 1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train,y_train)
# Set the check_input parameter to False to suppress the warning during prediction
rfc.check_input = False


@app.get('/')
def index():
    return {'message': 'Hello, welcome to Darb-API'}

@app.head("/items/{item_id}")
async def get_item_headers(item_id: int):
    # Do whatever processing you need for HEAD requests
    # In this example, we're just returning an empty response
    return {}

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

    prediction = rfc.predict([[preg,plas,pres,skin,insu,mass,pedi,age]])
    return{
        'prediction': str(prediction)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port= 8000)
else:
    uvicorn.run(app, host="0.0.0.0", port=8000)
