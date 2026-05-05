from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# load model
model = joblib.load('svm_balanced.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def preprocess_inference(text):
    # Cleaning
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    text = text.lower()

    return text

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(text: str = Form(...)):
    clean_text = preprocess_inference(text)
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector.toarray())[0]
    scores = model.decision_function(vector.toarray())[0]
    confidence = float(np.max(scores))

    return {
        "original": text,
        "clean": clean_text,
        "sentiment": prediction,
        "confidence": round(confidence, 2)
    }

# run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)