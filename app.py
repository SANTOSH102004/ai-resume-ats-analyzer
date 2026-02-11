import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document
import io

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_text(file):
    """
    Extract text from uploaded PDF or DOCX file.
    """
    text = ""
    if file.type == "application/pdf":
        try:
            # Try pdfplumber first
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except:
            # Fallback to PyPDF2
            reader = PdfReader(io.BytesIO(file.read()))
            for page in reader.pages:
                text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(file.read()))
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove special characters, stopwords, lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def calculate_similarity(resume_text, jd_text):
    """
    Calculate cosine similarity between resume and job description using TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100  # As percentage

def extract_keywords(jd_text, top_n=20):
    """
    Extract top keywords from job description using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keywords]

def generate_suggestions(matched_keywords, missing_keywords):
    """
    Generate suggestions based on matched and missing keywords.
    """
    suggestions = []
    if missing_keywords:
        suggestions.append(f"Add the following missing keywords to your resume: {', '.join(missing_keywords[:10])}")
    if len(matched_keywords) < 5:
        suggestions.append("Your resume matches few keywords. Consider tailoring it more to the job description.")
    else:
        suggestions.append("Good keyword match! Ensure your resume highlights these skills prominently.")
    suggestions.append("Use action verbs and quantify achievements where possible.")
    return suggestions

# Streamlit UI
st.set_page_config(page_title="AI Resume ATS Analyzer", page_icon="📄", layout="wide")

st.title("AI Resume ATS Analyzer")
st.markdown("Upload your resume and paste the job description to get an ATS compatibility score.")

# Upload section
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
with col2:
    job_description = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is not None and job_description.strip():
        with st.spinner("Analyzing..."):
            # Extract and preprocess
            resume_text = extract_text(uploaded_file)
            if not resume_text.strip():
                st.error("Could not extract text from the resume. Please check the file.")
                st.stop()
            processed_resume = preprocess_text(resume_text)
            processed_jd = preprocess_text(job_description)

            # Calculate ATS score
            ats_score = calculate_similarity(processed_resume, processed_jd)

            # Keyword analysis
            jd_keywords = extract_keywords(processed_jd)
            resume_words = set(processed_resume.split())
            matched_keywords = [kw for kw in jd_keywords if kw in resume_words]
            missing_keywords = [kw for kw in jd_keywords if kw not in resume_words]

            # Suggestions
            suggestions = generate_suggestions(matched_keywords, missing_keywords)

        # Display results
        st.success("Analysis Complete!")

        # ATS Score
        st.subheader("ATS Score")
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{ats_score:.1f}%</h1>", unsafe_allow_html=True)
        st.progress(ats_score / 100)

        # Score Breakdown
        st.subheader("Score Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similarity Score", f"{ats_score:.1f}%")
        with col2:
            match_percent = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
            st.metric("Keyword Match", f"{match_percent:.1f}%")
        with col3:
            st.metric("Improvement Potential", f"{100 - ats_score:.1f}%")

        # Keyword Analysis
        st.subheader("Keyword Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matched Keywords:**")
            if matched_keywords:
                st.write(", ".join(matched_keywords))
            else:
                st.write("None")
        with col2:
            st.write("**Missing Keywords:**")
            if missing_keywords:
                st.write(", ".join(missing_keywords))
            else:
                st.write("None")

        # Suggestions
        st.subheader("Suggestions for Improvement")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        # Download Report
        report = f"""
ATS Analysis Report

ATS Score: {ats_score:.1f}%

Matched Keywords: {', '.join(matched_keywords)}
Missing Keywords: {', '.join(missing_keywords)}

Suggestions:
{chr(10).join('- ' + s for s in suggestions)}
        """
        st.download_button("Download Report", report, file_name="ats_report.txt", mime="text/plain")

    else:
        st.error("Please upload a resume and enter a job description.")

# Sample Data
st.markdown("---")
st.subheader("Sample Data for Testing")
with st.expander("Sample Job Description"):
    st.code("""
We are looking for a Python Developer with experience in machine learning and data analysis.

Requirements:
- Proficiency in Python, scikit-learn, pandas, numpy
- Experience with NLP libraries like NLTK or spaCy
- Knowledge of web frameworks like Flask or Streamlit
- Strong problem-solving skills
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Develop and maintain ML models
- Analyze data and generate insights
- Collaborate with cross-functional teams
- Deploy models to production
    """, language="text")

with st.expander("Sample Resume Text"):
    st.code("""
John Doe
Python Developer

Skills:
- Python, Machine Learning, Data Analysis
- Libraries: scikit-learn, pandas, numpy, NLTK
- Frameworks: Flask, Django
- Tools: Git, Docker

Experience:
Software Engineer at XYZ Corp (2020-Present)
- Developed ML models using scikit-learn
- Built web apps with Flask
- Analyzed large datasets with pandas

Education:
B.S. Computer Science, University of ABC (2016-2020)
    """, language="text")
