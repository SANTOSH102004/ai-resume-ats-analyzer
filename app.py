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

def generate_suggestions(ats_score, matched_keywords, missing_keywords, resume_text, jd_text):
    """
    Generate detailed suggestions for resume improvement based on ATS score, keywords, and content analysis.
    """
    suggestions = []

    # Overall score-based suggestions
    if ats_score < 30:
        suggestions.append("Your resume has low compatibility. Consider a complete overhaul to better align with the job description.")
    elif ats_score < 50:
        suggestions.append("Your resume needs significant improvements. Focus on tailoring it specifically to this job.")
    elif ats_score < 70:
        suggestions.append("Your resume is moderately compatible. Minor adjustments can boost your score.")
    else:
        suggestions.append("Great job! Your resume is well-aligned. Fine-tune for even better results.")

    # Keyword-based suggestions
    if missing_keywords:
        suggestions.append(f"Incorporate these missing keywords into your resume: {', '.join(missing_keywords[:10])}. Place them naturally in relevant sections like skills, experience, or summary.")
    if len(matched_keywords) < 5:
        suggestions.append("Your resume matches few keywords from the job description. Tailor your content more closely to the job requirements.")
    else:
        suggestions.append("Good keyword match! Ensure these skills are prominently featured in your resume.")

    # Content-specific suggestions
    resume_words = resume_text.split()
    if len(resume_words) < 100:
        suggestions.append("Your resume is quite short. Expand on your experiences, skills, and achievements to provide more detail.")
    elif len(resume_words) > 600:
        suggestions.append("Your resume is lengthy. Consider condensing it to 1-2 pages, focusing on the most relevant information.")

    # General resume improvement tips
    suggestions.append("Use action verbs (e.g., 'developed', 'managed', 'optimized') at the beginning of bullet points.")
    suggestions.append("Quantify your achievements with numbers (e.g., 'Increased sales by 25%', 'Managed a team of 5').")
    suggestions.append("Customize your resume for each job application by incorporating job-specific keywords.")
    suggestions.append("Ensure your resume is ATS-friendly: use standard fonts, avoid tables/graphics, and save as PDF.")
    suggestions.append("Include a professional summary at the top highlighting your key qualifications.")
    suggestions.append("Proofread carefully for grammar and spelling errors.")

    # Job-specific suggestions based on JD content
    jd_lower = jd_text.lower()
    if 'experience' in jd_lower and 'year' in jd_lower:
        suggestions.append("Highlight your years of experience prominently in your resume.")
    if 'degree' in jd_lower or 'bachelor' in jd_lower:
        suggestions.append("Clearly list your educational qualifications and degrees.")
    if 'certification' in jd_lower:
        suggestions.append("Include relevant certifications if you have them, or consider obtaining them.")

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
            suggestions = generate_suggestions(ats_score, matched_keywords, missing_keywords, processed_resume, processed_jd)

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
