from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import pickle
import numpy as np
import pandas as pd
import Classifier

app = FastAPI()
origins = [
    'http://localhost:8060',
]
# class CustomUnpickler(pickle.Unpickler):

#     def find_class(self, module, name):
#         if name == 'base_model.pkl':
#             from Classifier import Classifier
#             return Classifier
#         return super().find_class(module, name)

# current_model = 'base_model.pkl'

# model = CustomUnpickler(open(current_model, 'rb')).load()

# app.middleware(
#     CORSMiddleware,
#     allow_origins = origins,
#     allow_credentials = True,
#     allow_methods = ['*'],
#     allow_headers = ['*']
# )

model = Classifier.Classifier(1024, [32,32,10])
filename = 'base_model_weights.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
model.load_model(loaded_model)

@app.get("/")
async def root():
    return {"message": "Wrong Method"}

@app.post("/")
async def upload(file: bytes = File(...)):
    # print(file)
    image = Image.open(io.BytesIO(file))
    # image= image.convert('L')
    image = np.array(image)
    
    print(image.shape)
    image.resize((32,32))
    image = image/255
    # image_re.show()
    image = image.reshape(-1,)
    # print(list(image))
    print(image.shape)
    result = model.predict(image)
    print("The result is :", result)
    return {"Prediction": str(result), 'status': 200}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8060)
    