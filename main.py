from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"status": "API is running"}

class Email(BaseModel):
    text: str

@app.post("/classify")
def classify(email: Email):
    try:
        prediction = model.predict([email.text])[0]
        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}
