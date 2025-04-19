import os
import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
from datetime import datetime
import subprocess
import sys

# Disable GPU and TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Install en_core_web_sm if not present
@st.cache_resource
def install_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        return spacy.load("en_core_web_sm")
    return spacy.load("en_core_web_sm")

# Initialize models (use distilgpt2 for lighter footprint)
@st.cache_resource
def load_models():
    try:
        generator = pipeline("text-generation", model="distilgpt2", device=-1)  # Explicitly use CPU
        nlp = install_spacy_model()
        return generator, nlp
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

generator, nlp = load_models()

# Check if models loaded successfully
if generator is None or nlp is None:
    st.stop()

# Simulated employee data (replace with HR API in production)
default_data = {
    "employee_id": "E123",
    "name": "Keni Sharma",
    "role": "Software Engineer",
    "kpis": {"Project Completion": 95, "Team Collaboration": 80},
    "feedback": ["Great work on project X", "Needs to improve meeting participation"],
    "goals": ["Complete AI course by Q4", "Lead a team project"]
}

# Function to generate performance review
def generate_review(data):
    kpis_str = ", ".join([f"{k}: {v}%" for k, v in data["kpis"].items()])
    feedback_str = ", ".join(data["feedback"])
    prompt = f"""
    Write a concise, professional performance review for {data['name']}, a {data['role']}.
    KPIs: {kpis_str}
    Feedback: {feedback_str}
    Highlight strengths, areas for improvement, and align with company goals.
    Keep it under 150 words.
    """
    try:
        review = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]["generated_text"]
        return review.split("\n")[0]  # Simplify output for demo
    except Exception as e:
        return f"Error generating review: {str(e)}"

# Function to analyze feedback sentiment and detect bias
def analyze_feedback(feedback):
    try:
        doc = nlp(feedback)
        sentiment_score = sum([token.sentiment for token in doc]) / len(doc) if doc else 0
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        # Simplified bias detection (expand in production)
        biased_terms = [token.text for token in doc if token.text.lower() in ["gender", "race", "aggressive"]]
        return {"sentiment": sentiment, "biased_terms": biased_terms}
    except Exception as e:
        return {"sentiment": "Error", "biased_terms": [f"Error: {str(e)}"]}

# Function to generate SMART goals
def generate_smart_goals(data):
    kpis_str = ", ".join([f"{k}: {v}%" for k, v in data["kpis"].items()])
    feedback_str = ", ".join(data["feedback"])
    prompt = f"""
    Generate 2 concise SMART goals for {data['name']}, a {data['role']}.
    KPIs: {kpis_str}
    Feedback: {feedback_str}
    Align with career growth and company objectives.
    Keep each goal under 50 words.
    """
    try:
        goals = generator(prompt, max_length=150, num_return_sequences=2, truncation=True)
        return [goal["generated_text"].split("\n")[0] for goal in goals]
    except Exception as e:
        return [f"Error generating goal: {str(e)}"]

# Streamlit app
st.title("Performance Management Automation Prototype")
st.write("Demonstrates AI-powered performance reviews, feedback analysis, and goal tracking.")

# Input form
st.header("Enter Employee Data")
with st.form("employee_form"):
    name = st.text_input("Employee Name", value=default_data["name"])
    role = st.text_input("Role", value=default_data["role"])
    kpi1_value = st.number_input("Project Completion (%)", min_value=0, max_value=100, value=default_data["kpis"]["Project Completion"])
    kpi2_value = st.number_input("Team Collaboration (%)", min_value=0, max_value=100, value=default_data["kpis"]["Team Collaboration"])
    feedback1 = st.text_area("Feedback 1", value=default_data["feedback"][0])
    feedback2 = st.text_area("Feedback 2", value=default_data["feedback"][1])
    submitted = st.form_submit_button("Generate Outputs")

# Process data and display results
if submitted:
    # Prepare data
    employee_data = {
        "employee_id": "E123",
        "name": name,
        "role": role,
        "kpis": {"Project Completion": kpi1_value, "Team Collaboration": kpi2_value},
        "feedback": [feedback1, feedback2],
        "goals": []
    }

    # Generate review
    st.header("Performance Review")
    with st.spinner("Generating review..."):
        review = generate_review(employee_data)
    st.write(review)

    # Analyze feedback
    st.header("Feedback Analysis")
    for fb in employee_data["feedback"]:
        analysis = analyze_feedback(fb)
        st.subheader(f"Feedback: {fb}")
        st.write(f"Sentiment: {analysis['sentiment']}")
        st.write(f"Biased Terms: {', '.join(analysis['biased_terms']) or 'None'}")

    # Generate goals
    st.header("SMART Goals")
    with st.spinner("Generating goals..."):
        goals = generate_smart_goals(employee_data)
    for i, goal in enumerate(goals, 1):
        st.write(f"Goal {i}: {goal}")
