from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Load the saved model and the TF-IDF vectorizer
clf = joblib.load("M1.pkl")
tfidf = joblib.load("tfidf1.pkl")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],   # Allow all headers
)

# Define the data model for input (list of skills)
class SkillsInput(BaseModel):
    skills: list[str]

# Prediction function
def predict_job_title(skills_list):
    skills_str = " ".join(skills_list)  # Join the skills into a single string
    tagline_tfidf = tfidf.transform([skills_str])  # Transform the input using TF-IDF
    predicted_job_title = clf.predict(tagline_tfidf)  # Predict using the model
    return predicted_job_title[0]

# Define a POST endpoint that takes input from frontend
@app.post("/predict-job-title/")
async def get_job_title(input_data: SkillsInput):
    skills_list = input_data.skills  # Extract the skills from the request body
    predicted_title = predict_job_title(skills_list)  # Predict the job title
    return {"predicted_job_title": predicted_title}
