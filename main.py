from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import pickle
import numpy as np
import pandas as pd
import uvicorn
from API.Classifier import Classifier

app = FastAPI()
origins = [
    'http://localhost:8000',
]
model = Classifier(1024, [32,32,10])
filename = 'base_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model.weights1)
result = loaded_model.predict(X_test)

@app.get("/")
async def root():
    return {"message": "Wrong Method"}

@app.post("/image")
async def upload(file: bytes = File(...)):
    print(result)
    image = Image.open(io.BytesIO(file))
#     image.show()
    image = np.array(image)
#     image = image.resize((32, 32))
#     image = image.reshape(-1,)
    print(image.shape)
    # result = model.predict(image)
    print("The result is :", result)
    return {"Upload Status": "Complete"}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8060)