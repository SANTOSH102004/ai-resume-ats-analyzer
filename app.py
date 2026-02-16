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

# Industry categories for analysis
INDUSTRY_TEMPLATES = {
    "Technology": {
        "keywords": ["python", "java", "javascript", "react", "angular", "vue", "node", "sql", "database", "cloud", "aws", "azure", "docker", "kubernetes", "git", "agile", "scrum", "machine learning", "deep learning", "ai", "data science"],
        "sections": ["Skills", "Experience", "Projects", "Education", "Certifications"]
    },
    "Finance": {
        "keywords": ["accounting", "financial analysis", "excel", "sql", "reporting", "budgeting", "forecasting", "risk management", "audit", "tax", "gaap", "ifrs", "financial modeling"],
        "sections": ["Summary", "Experience", "Education", "Certifications", "Skills"]
    },
    "Healthcare": {
        "keywords": ["patient care", "clinical", "healthcare", "medical", "nursing", "pharmaceutical", "research", "compliance", "hipaa", "electronic health records", "diagnosis", "treatment"],
        "sections": ["Clinical Experience", "Education", "Certifications", "Skills", "Professional Memberships"]
    },
    "Marketing": {
        "keywords": ["digital marketing", "seo", "sem", "social media", "content marketing", "email marketing", "analytics", "google ads", "facebook ads", "brand management", "marketing strategy", "campaign"],
        "sections": ["Summary", "Experience", "Skills", "Education", "Portfolio"]
    },
    "Engineering": {
        "keywords": ["cad", "solidworks", "autocad", "matlab", "simulink", "project management", "manufacturing", "design", "testing", "quality assurance", "mechanical", "electrical", "civil"],
        "sections": ["Technical Skills", "Experience", "Projects", "Education", "Certifications"]
    },
    "General": {
        "keywords": ["communication", "leadership", "teamwork", "problem solving", "time management", "analytical", "organizational"],
        "sections": ["Summary", "Experience", "Skills", "Education"]
    }
}

# Skill categories dictionary
SKILL_CATEGORIES = {
    "programming_languages": ["python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "typescript", "php", "swift", "kotlin", "scala", "r"],
    "frameworks": ["react", "angular", "vue", "django", "flask", "spring", "node.js", "express", "next.js", "laravel", "rails", ".net"],
    "databases": ["sql", "mysql", "postgresql", "mongodb", "oracle", "redis", "elasticsearch", "cassandra", "firebase"],
    "cloud_devops": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "ci/cd", "git", "jira"],
    "data_science": ["machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark", "tableau", "power bi", "nlp", "computer vision"],
    "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "analytical", "time management", "adaptability", "creativity", "conflict resolution"]
}

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

def analyze_skills(resume_text, jd_text):
    """
    Analyze and categorize skills from resume and job description.
    Returns categorized skills: technical, soft, and shows matched/missing skills.
    """
    # Preprocess texts
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Categorize JD skills
    jd_technical = []
    jd_soft = []
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill in jd_lower:
                if category == "soft_skills":
                    jd_soft.append(skill)
                else:
                    jd_technical.append(skill)
    
    # Find matched and missing skills
    matched_technical = [s for s in jd_technical if s in resume_lower]
    missing_technical = [s for s in jd_technical if s not in resume_lower]
    matched_soft = [s for s in jd_soft if s in resume_lower]
    missing_soft = [s for s in jd_soft if s not in resume_lower]
    
    return {
        "technical": {
            "matched": matched_technical,
            "missing": missing_technical,
            "jd_skills": jd_technical
        },
        "soft": {
            "matched": matched_soft,
            "missing": missing_soft,
            "jd_skills": jd_soft
        }
    }

def check_resume_format(resume_text, file_type):
    """
    Check resume formatting issues.
    """
    issues = []
    suggestions = []
    
    # Check length
    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append("Resume is too short")
        suggestions.append("Add more details about your experience and achievements")
    elif word_count > 1500:
        issues.append("Resume is too long")
        suggestions.append("Consider condensing to 1-2 pages, focusing on relevant information")
    
    # Check for common sections
    required_sections = ["experience", "education", "skills"]
    found_sections = []
    for section in required_sections:
        if section in resume_text.lower():
            found_sections.append(section)
    
    if len(found_sections) < 2:
        issues.append("Missing important sections")
        suggestions.append("Ensure your resume has clear Experience, Education, and Skills sections")
    
    # Check for action verbs
    action_verbs = ["developed", "managed", "led", "created", "implemented", "designed", "analyzed", "increased", "reduced", "optimized"]
    has_action_verbs = any(verb in resume_text.lower() for verb in action_verbs)
    if not has_action_verbs:
        issues.append("Limited use of action verbs")
        suggestions.append("Start bullet points with strong action verbs (e.g., Developed, Managed, Led)")
    
    # Check for quantifiable achievements
    has_numbers = any(char.isdigit() for char in resume_text)
    if not has_numbers:
        issues.append("No quantifiable achievements")
        suggestions.append("Add numbers to demonstrate impact (e.g., 'Increased sales by 25%')")
    
    # File type check
    if file_type == "application/pdf":
        suggestions.append("PDF format is ATS-friendly ✓")
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        suggestions.append(" DOCX format may have compatibility issues with some ATS systems")
    
    return {
        "issues": issues,
        "suggestions": suggestions,
        "word_count": word_count
    }

def extract_experience_years(text):
    """
    Extract years of experience mentioned in the text.
    """
    # Look for patterns like "X years", "X+ years", etc.
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*year\s+(?:professional|work)',
    ]
    
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years_found.extend([int(y) for y in matches])
    
    return max(years_found) if years_found else None

def analyze_experience(resume_text, jd_text):
    """
    Compare years of experience required vs provided.
    """
    # Extract required years from JD
    jd_years = extract_experience_years(jd_text)
    resume_years = extract_experience_years(resume_text)
    
    # Also check for experience mentions
    jd_experience_mentions = len(re.findall(r'experience', jd_text.lower()))
    resume_experience_mentions = len(re.findall(r'experience', resume_text.lower()))
    
    result = {
        "jd_required_years": jd_years,
        "resume_years": resume_years,
        "meets_requirement": None,
        "analysis": []
    }
    
    if jd_years:
        if resume_years:
            if resume_years >= jd_years:
                result["meets_requirement"] = True
                result["analysis"].append(f"You have {resume_years} years of experience, meeting the {jd_years}+ years requirement ✓")
            else:
                result["meets_requirement"] = False
                result["analysis"].append(f"You have {resume_years} years of experience, but the job requires {jd_years}+ years")
        else:
            result["analysis"].append("Could not determine years of experience from resume. Clearly list your work history with dates.")
    
    if jd_experience_mentions > 0 and resume_experience_mentions < jd_experience_mentions:
        result["analysis"].append("The job description mentions more experience areas. Ensure all relevant experience is clearly documented.")
    
    return result

def analyze_education(resume_text, jd_text):
    """
    Check if education requirements are met.
    """
    # Education levels
    education_levels = {
        "high school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5,
        "doctorate": 5,
        "mba": 4
    }
    
    # Find required education in JD
    jd_education = []
    for edu, level in education_levels.items():
        if edu in jd_text.lower():
            jd_education.append((edu, level))
    
    # Find education in resume
    resume_education = []
    for edu, level in education_levels.items():
        if edu in resume_text.lower():
            resume_education.append((edu, level))
    
    result = {
        "required": jd_education,
        "found": resume_education,
        "meets_requirement": None,
        "analysis": []
    }
    
    if jd_education:
        required_level = max([level for _, level in jd_education])
        
        if resume_education:
            found_level = max([level for _, level in resume_education])
            if found_level >= required_level:
                result["meets_requirement"] = True
                result["analysis"].append(f"You meet the education requirement ✓")
            else:
                result["meets_requirement"] = False
                result["analysis"].append(f"The job requires {jd_education[0][0]} level, but your resume shows {resume_education[0][0]}")
        else:
            result["analysis"].append("Could not clearly identify education level in resume. Ensure degrees are prominently listed.")
    
    return result

def detect_industry(jd_text):
    """
    Detect the most likely industry from job description.
    """
    jd_lower = jd_text.lower()
    scores = {}
    
    for industry, data in INDUSTRY_TEMPLATES.items():
        score = sum(1 for keyword in data["keywords"] if keyword in jd_lower)
        scores[industry] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "General"

def get_industry_advice(industry, resume_sections):
    """
    Get industry-specific resume advice.
    """
    if industry not in INDUSTRY_TEMPLATES:
        return ["Follow standard resume formatting for best ATS compatibility."]
    
    template = INDUSTRY_TEMPLATES[industry]
    advice = []
    
    # Check for recommended sections
    found_sections = [s.lower() for s in resume_sections]
    recommended = template.get("sections", [])
    
    missing_sections = [s for s in recommended if s.lower() not in found_sections]
    if missing_sections:
        advice.append(f"Consider adding these sections for {industry} roles: {', '.join(missing_sections[:3])}")
    
    # Industry-specific tips
    industry_tips = {
        "Technology": "Highlight technical skills, projects, and certifications. Include GitHub/portfolio links.",
        "Finance": "Emphasize certifications (CPA, CFA), specific software proficiency, and quantifiable financial achievements.",
        "Healthcare": "Include relevant licenses, certifications, and clinical experience. Highlight patient care skills.",
        "Marketing": "Showcase marketing campaigns, analytics tools, and creative portfolio. Include metrics.",
        "Engineering": "Detail specific software (CAD, MATLAB), projects, and technical certifications."
    }
    
    if industry in industry_tips:
        advice.append(industry_tips[industry])
    
    return advice

def analyze_cover_letter(cover_letter_text, resume_text, jd_text):
    """
    Analyze cover letter compatibility with resume and job description.
    """
    if not cover_letter_text or not cover_letter_text.strip():
        return None
    
    # Preprocess texts
    cl_processed = preprocess_text(cover_letter_text)
    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(jd_text)
    
    # Calculate similarities
    cl_jd_similarity = calculate_similarity(cl_processed, jd_processed)
    cl_resume_similarity = calculate_similarity(cl_processed, resume_processed)
    
    # Extract keywords from cover letter
    cl_keywords = extract_keywords(cover_letter_text, top_n=10)
    jd_keywords = extract_keywords(jd_text, top_n=20)
    
    matched_cl_keywords = [kw for kw in cl_keywords if kw in jd_keywords]
    
    analysis = {
        "similarity_with_jd": cl_jd_similarity,
        "similarity_with_resume": cl_resume_similarity,
        "matched_keywords": matched_cl_keywords,
        "suggestions": []
    }
    
    # Suggestions
    if cl_jd_similarity < 40:
        analysis["suggestions"].append("Your cover letter should better address the job requirements mentioned in the job description.")
    
    if cl_resume_similarity < 30:
        analysis["suggestions"].append("Ensure your cover letter complements your resume rather than repeating it.")
    
    if len(matched_cl_keywords) < 3:
        analysis["suggestions"].append("Incorporate more job-specific keywords into your cover letter.")
    
    if not analysis["suggestions"]:
        analysis["suggestions"].append("Good cover letter alignment! Make sure it's concise and engaging.")
    
    return analysis

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
st.markdown("Upload your resume and paste the job description to get a comprehensive ATS compatibility analysis.")

# Upload section with optional cover letter
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    # Optional: Cover letter upload
    cover_letter_file = st.file_uploader("Upload Cover Letter (Optional - PDF or DOCX)", type=["pdf", "docx"])
with col2:
    job_description = st.text_area("Paste Job Description", height=200)

# Industry selector (auto-detected but can be overridden)
industry = st.selectbox("Industry (Auto-detected from JD)", 
                        options=["Auto-Detect"] + list(INDUSTRY_TEMPLATES.keys()))

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is not None and job_description.strip():
        with st.spinner("Analyzing..."):
            # Extract and preprocess
            resume_text = extract_text(uploaded_file)
            if not resume_text.strip():
                st.error("Could not extract text from the resume. Please check the file.")
                st.stop()
            
            # Process cover letter if provided
            cover_letter_text = ""
            if cover_letter_file:
                cover_letter_text = extract_text(cover_letter_file)
            
            processed_resume = preprocess_text(resume_text)
            processed_jd = preprocess_text(job_description)

            # Calculate ATS score
            ats_score = calculate_similarity(processed_resume, processed_jd)

            # Keyword analysis
            jd_keywords = extract_keywords(processed_jd)
            resume_words = set(processed_resume.split())
            matched_keywords = [kw for kw in jd_keywords if kw in resume_words]
            missing_keywords = [kw for kw in jd_keywords if kw not in resume_words]

            # ====== NEW FEATURES ======
            
            # 1. Skills Analysis
            skills_analysis = analyze_skills(resume_text, job_description)
            
            # 2. Resume Format Checker
            format_analysis = check_resume_format(resume_text, uploaded_file.type)
            
            # 3. Experience Matching
            experience_analysis = analyze_experience(resume_text, job_description)
            
            # 4. Education Qualifier
            education_analysis = analyze_education(resume_text, job_description)
            
            # 7. Industry-Specific Templates
            detected_industry = detect_industry(job_description)
            if industry == "Auto-Detect":
                selected_industry = detected_industry
            else:
                selected_industry = industry
            
            # Get resume sections for industry advice
            resume_sections = re.findall(r'^[A-Z][a-z]+:', resume_text, re.MULTILINE)
            industry_advice = get_industry_advice(selected_industry, resume_sections)
            
            # 8. Cover Letter Analysis
            cover_letter_analysis = None
            if cover_letter_text:
                cover_letter_analysis = analyze_cover_letter(cover_letter_text, resume_text, job_description)

            # Suggestions
            suggestions = generate_suggestions(ats_score, matched_keywords, missing_keywords, processed_resume, processed_jd)

        # Display results
        st.success("Analysis Complete!")
        
        # Industry Detection
        if industry == "Auto-Detect":
            st.info(f"📊 Detected Industry: **{detected_industry}**")

        # ATS Score
        st.subheader("📈 ATS Score")
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{ats_score:.1f}%</h1>", unsafe_allow_html=True)
        st.progress(ats_score / 100)

        # Score Breakdown
        st.subheader("📊 Score Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Similarity Score", f"{ats_score:.1f}%")
        with col2:
            match_percent = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
            st.metric("Keyword Match", f"{match_percent:.1f}%")
        with col3:
            st.metric("Improvement Potential", f"{100 - ats_score:.1f}%")
        with col4:
            st.metric("Word Count", format_analysis["word_count"])

        # ====== Skills Analysis Section ======
        st.markdown("---")
        st.subheader("🔧 Skills Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Technical Skills Matched:**")
            if skills_analysis["technical"]["matched"]:
                st.success(", ".join(skills_analysis["technical"]["matched"]))
            else:
                st.warning("No technical skills matched")
            
            st.write("**Technical Skills Missing:**")
            if skills_analysis["technical"]["missing"]:
                st.error(", ".join(skills_analysis["technical"]["missing"]))
            else:
                st.success("None - Great job!")
        
        with col2:
            st.write("**Soft Skills Matched:**")
            if skills_analysis["soft"]["matched"]:
                st.success(", ".join(skills_analysis["soft"]["matched"]))
            else:
                st.warning("No soft skills matched")
            
            st.write("**Soft Skills Missing:**")
            if skills_analysis["soft"]["missing"]:
                st.error(", ".join(skills_analysis["soft"]["missing"]))
            else:
                st.success("None - Great job!")

        # ====== Experience Analysis Section ======
        st.markdown("---")
        st.subheader("💼 Experience Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if experience_analysis["jd_required_years"]:
                st.metric("Required Experience", f"{experience_analysis['jd_required_years']}+ years")
            else:
                st.metric("Required Experience", "Not specified")
        
        with col2:
            if experience_analysis["resume_years"]:
                st.metric("Your Experience", f"{experience_analysis['resume_years']} years")
            else:
                st.metric("Your Experience", "Not detected")
        
        if experience_analysis["meets_requirement"] is not None:
            if experience_analysis["meets_requirement"]:
                st.success("✓ You meet the experience requirement!")
            else:
                st.warning("⚠ You may not meet the experience requirement")
        
        for analysis in experience_analysis["analysis"]:
            st.write(f"- {analysis}")

        # ====== Education Analysis Section ======
        st.markdown("---")
        st.subheader("🎓 Education Analysis")
        
        if education_analysis["required"]:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Required Education:**")
                for edu, _ in education_analysis["required"]:
                    st.write(f"- {edu.title()}")
            
            with col2:
                st.write("**Your Education:**")
                if education_analysis["found"]:
                    for edu, _ in education_analysis["found"]:
                        st.write(f"- {edu.title()}")
                else:
                    st.warning("Not detected in resume")
            
            if education_analysis["meets_requirement"] is not None:
                if education_analysis["meets_requirement"]:
                    st.success("✓ You meet the education requirement!")
                else:
                    st.warning("⚠ You may not meet the education requirement")
        
        for analysis in education_analysis["analysis"]:
            st.write(f"- {analysis}")

        # ====== Format Checker Section ======
        st.markdown("---")
        st.subheader("📋 Resume Format Checker")
        
        if format_analysis["issues"]:
            for issue in format_analysis["issues"]:
                st.warning(f"⚠ {issue}")
        
        for suggestion in format_analysis["suggestions"]:
            if "✓" in suggestion:
                st.success(f"✓ {suggestion}")
            else:
                st.info(f"💡 {suggestion}")

        # ====== Industry-Specific Advice ======
        st.markdown("---")
        st.subheader(f"🏢 Industry-Specific Advice ({selected_industry})")
        
        for advice in industry_advice:
            st.write(f"- {advice}")

        # ====== Cover Letter Analysis ======
        if cover_letter_analysis:
            st.markdown("---")
            st.subheader("✉ Cover Letter Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CL-JD Similarity", f"{cover_letter_analysis['similarity_with_jd']:.1f}%")
            with col2:
                st.metric("CL-Resume Similarity", f"{cover_letter_analysis['similarity_with_resume']:.1f}%")
            
            st.write("**Matched Keywords in Cover Letter:**")
            if cover_letter_analysis["matched_keywords"]:
                st.success(", ".join(cover_letter_analysis["matched_keywords"]))
            else:
                st.warning("No keywords matched")
            
            for suggestion in cover_letter_analysis["suggestions"]:
                st.write(f"- {suggestion}")

        # ====== Keyword Analysis ======
        st.markdown("---")
        st.subheader("🔑 Keyword Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matched Keywords:**")
            if matched_keywords:
                st.success(", ".join(matched_keywords))
            else:
                st.write("None")
        with col2:
            st.write("**Missing Keywords:**")
            if missing_keywords:
                st.error(", ".join(missing_keywords))
            else:
                st.write("None")

        # ====== Suggestions ======
        st.markdown("---")
        st.subheader("💡 Suggestions for Improvement")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        # Download Report
        report = f"""
ATS Analysis Report
==================

ATS Score: {ats_score:.1f}%

INDUSTRY: {selected_industry}

SKILLS ANALYSIS:
- Technical Skills Matched: {', '.join(skills_analysis['technical']['matched']) if skills_analysis['technical']['matched'] else 'None'}
- Technical Skills Missing: {', '.join(skills_analysis['technical']['missing']) if skills_analysis['technical']['missing'] else 'None'}
- Soft Skills Matched: {', '.join(skills_analysis['soft']['matched']) if skills_analysis['soft']['matched'] else 'None'}
- Soft Skills Missing: {', '.join(skills_analysis['soft']['missing']) if skills_analysis['soft']['missing'] else 'None'}

EXPERIENCE ANALYSIS:
{chr(10).join('- ' + a for a in experience_analysis['analysis'])}

EDUCATION ANALYSIS:
{chr(10).join('- ' + a for a in education_analysis['analysis'])}

FORMAT CHECKER:
- Word Count: {format_analysis['word_count']}
{chr(10).join('- ' + i for i in format_analysis['issues']) if format_analysis['issues'] else '- No major format issues'}

KEYWORDS:
- Matched: {', '.join(matched_keywords)}
- Missing: {', '.join(missing_keywords)}

SUGGESTIONS:
{chr(10).join('- ' + s for s in suggestions)}
        """
        
        if cover_letter_analysis:
            report += f"""

COVER LETTER ANALYSIS:
- CL-JD Similarity: {cover_letter_analysis['similarity_with_jd']:.1f}%
- CL-Resume Similarity: {cover_letter_analysis['similarity_with_resume']:.1f}%
- Matched Keywords: {', '.join(cover_letter_analysis['matched_keywords'])}
"""
        
        st.download_button("Download Full Report", report, file_name="ats_full_report.txt", mime="text/plain")

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
- 3+ years of experience in software development
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
