from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    g1_1: float
    g1_2: float
    g1_3: float
    g1_4: float
    g1_5: float
    g1_6: float
    g1_7: float
    g1_8: float
    g1_9: float
    g1_10: float
    g1_11: float
    g1_12: float
    g1_13: float
    g1_14: float
    g1_15: float
    g1_16: float
    g1_17: float
    g1_18: float
    g1_19: float
    g1_20: float
    g2_1: float
    g2_2: float
    g2_3: float
    g2_4: float
    g2_5: float
    g2_6: float
    g2_7: float
    g2_8: float
    g2_9: float
    g2_10: float
    g2_11: float
    g2_12: float
    g2_13: float
    g2_14: float
    g2_15: float
    g2_16: float
    g2_17: float
    g2_18: float
    g2_19: float
    g2_20: float
    c_1: float
    c_2: float
    c_3: float
    c_4: float
    c_5: float
    c_6: float
    c_7: float
    c_8: float
    c_9: float
    c_10: float
    c_11: float
    c_12: float
    c_13: float
    c_14: float
    c_15: float
    c_16: float
    c_17: float
    c_18: float
    c_19: float
    c_20: float
    c_21: float
    c_22: float
    c_23: float
    c_24: float
    c_25: float
    c_26: float
    c_27: float

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item: Item):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    y_pred = model.predict(df)
    return {"prediction":int(y_pred)}