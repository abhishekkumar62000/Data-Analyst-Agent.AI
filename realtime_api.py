from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import time

app = FastAPI()

@app.get("/realtime-data")
def get_data():
    # Random data, replace with your logic
    return JSONResponse({"value": random.randint(1, 100), "timestamp": time.time()})
