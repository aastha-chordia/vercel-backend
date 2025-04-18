from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# CORS to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("reddit_comments_sorted_by_coin.csv")

@app.get("/")
def root():
    return {"message": "Hello from the backend!"}

@app.get("/comments/{coin}")
def get_comments(coin: str):
    filtered = df[df["coin"].str.lower() == coin.lower()]
    return filtered.to_dict(orient="records")
