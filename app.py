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
import os
import sys
import datetime
from anthropic import Anthropic
try:
    import ollama
except ImportError:
    ollama = None

# Initialize NLTK resources with caching
@st.cache_resource
def initialize_nltk_resources():
    """
    Download and initialize NLTK resources.
    Cached to avoid redownloading on every app rerun.
    """
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
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

# Initialize lemmatizer and stopwords
lemmatizer, stop_words = initialize_nltk_resources()

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
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving", 
        "analytical", "time management", "adaptability", "creativity", 
        "conflict resolution", "collaboration", "coordination", "interpersonal",
        "multitasking", "attention to detail", "critical thinking",
        "decision making", "presentation", "negotiation", "mentoring"
    ]
}

# Synonym mapping for soft skills - helps recognize related words in resumes
SOFT_SKILL_SYNONYMS = {
    "communication": ["communication", "communicated", "presented", "public speaking", "speaking"],
    "teamwork": ["teamwork", "team", "collaborated", "cross-functional", "coordination", "cooperate", "cooperative"],
    "leadership": ["leadership", "led", "leader", "managed", "mentored", "guided", "directing", "supervising"],
    "problem solving": ["problem solving", "problem-solving", "troubleshoot", "troubleshooting", "resolved", "analytical", "solution-oriented"],
    "adaptability": ["adaptability", "adaptive", "flexible", "flexibility", "dynamic", "versatile"],
    "creative": ["creativity", "creative", "innovative", "innovation"],
    "attention to detail": ["attention to detail", "detail-oriented", "meticulous", "thorough"],
    "critical thinking": ["critical thinking", "think critically", "analytical thinking"],
    "decision making": ["decision making", "decision-making", "decisive"],
    "collaboration": ["collaboration", "collaborate", "collaborative", "team collaboration"],
    "interpersonal": ["interpersonal", "interpersonal skills", "people skills", "soft skills"]
}


def extract_text(file):
    """
    Extract text from uploaded PDF or DOCX file.
    """
    text = ""
    if file.type == "application/pdf":
        # Read file bytes once to avoid stream exhaustion
        file_bytes = file.read()
        try:
            # Try pdfplumber first
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            # Fallback to PyPDF2
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(file.read()))
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

@st.cache_data
def preprocess_text(text):
    """
    Preprocess text: lowercase, remove special characters, stopwords, lemmatize.
    Cached to avoid reprocessing the same text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def calculate_similarity(resume_text, jd_text):
    """
    Calculate cosine similarity between resume and job description using TF-IDF.
    Cached to avoid recalculating for the same text pairs.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100  # As percentage

def extract_keywords(jd_text, top_n=20):
    """
    Extract top keywords from job description using TF-IDF.
    Supports both single words and multi-word phrases (bigrams).
    """
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([jd_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keywords]

def skill_in_text(skill, text):
    """
    Check if a skill exists in text with intelligent matching.
    Handles skill aliases (e.g., "javascript" matches "JS", "React" matches "React.js")
    and prevents false positives from ambiguous short skills.
    
    Args:
        skill: Skill name to search for
        text: Text to search in
        
    Returns:
        bool: True if skill found, False otherwise
    """
    text_lower = text.lower()
    skill_lower = skill.lower()
    
    # Handle special skill aliases and blacklist entries
    skill_aliases = {
        "javascript": ["javascript", "js"],
        "react": ["react", "react.js", "reactjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "node": ["node.js", "nodejs", "node"],
        "vue": ["vue", "vue.js", "vuejs"],
        "vue.js": ["vue", "vue.js", "vuejs"],
        "next.js": ["next.js", "nextjs"],
        "next": ["next.js", "nextjs"],
        "go": ["golang"],  # "go" alone is too ambiguous, only match "golang"
        "r": [],  # remove "r" entirely - too ambiguous
        "c++": ["c++", "cpp"],
        "c#": ["c#", "csharp"],
        "sql": ["sql", "mysql", "postgresql", "mssql"],
    }
    
    # If skill has aliases defined, check those instead
    if skill_lower in skill_aliases:
        aliases = skill_aliases[skill_lower]
        if not aliases:
            return False  # blacklisted skills like "r"
        return any(alias in text_lower for alias in aliases)
    
    # Default: word boundary match
    pattern = r'\b' + re.escape(skill_lower) + r'\b'
    return bool(re.search(pattern, text_lower, re.IGNORECASE))

def soft_skill_in_text(skill, text):
    """
    Check if a soft skill exists in text, including synonyms.
    
    Args:
        skill: Soft skill name
        text: Text to search in (should be lowercased)
        
    Returns:
        bool: True if skill or any of its synonyms are found
    """
    # First check exact match
    if skill_in_text(skill, text):
        return True
    
    # Then check synonyms if available
    if skill in SOFT_SKILL_SYNONYMS:
        for synonym in SOFT_SKILL_SYNONYMS[skill]:
            if skill_in_text(synonym, text):
                return True
    
    return False

def analyze_skills(resume_text, jd_text):
    """
    Analyze and categorize skills from resume and job description.
    Returns categorized skills: technical, soft, and shows matched/missing skills.
    Uses word-boundary matching and synonym detection for soft skills.
    """
    # Preprocess texts
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Categorize JD skills
    jd_technical = []
    jd_soft = []
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill_in_text(skill, jd_lower):
                if category == "soft_skills":
                    jd_soft.append(skill)
                else:
                    jd_technical.append(skill)
    
    # Find matched and missing skills using word boundary matching for technical
    matched_technical = [s for s in jd_technical if skill_in_text(s, resume_lower)]
    missing_technical = [s for s in jd_technical if not skill_in_text(s, resume_lower)]
    
    # For soft skills, check exact match AND synonyms
    matched_soft = [s for s in jd_soft if soft_skill_in_text(s, resume_lower)]
    missing_soft = [s for s in jd_soft if not soft_skill_in_text(s, resume_lower)]
    
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

def find_resume_sections(text):
    """
    Find resume sections regardless of casing or formatting.
    Handles extra whitespace, newlines, ALL CAPS headers, and various section names.
    
    Args:
        text: Resume text to analyze
        
    Returns:
        list: Found section names (lowercased)
    """
    # Normalize whitespace to handle DOCX formatting and multi-line headers
    text_lower = ' '.join(text.lower().split())
    
    # Map core sections to their common variations (including ALL CAPS variants)
    section_keywords = {
        "experience": ["experience", "work history", "employment history", 
                       "professional experience", "internship", "work experience", "career history"],
        "education": ["education", "academic background", "qualification", 
                      "educational background", "degrees", "academic"],
        "skills": ["skills", "technical skills", "core competencies", 
                   "competencies", "expertise", "technologies", "technical expertise"],
        "summary": ["summary", "professional summary", "objective", 
                    "career objective", "profile", "about me", "professional profile"],
        "certifications": ["certification", "certifications", "certificate", "licenses", "professional certification"],
        "projects": ["project", "projects", "academic projects", "key projects", "project experience"]
    }
    
    found = []
    for section, keywords in section_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found.append(section)
    
    return found

def check_resume_format(resume_text, file_type):
    """
    Check resume formatting issues.
    Uses robust section detection to handle various formatting and casing.
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
    
    # Check for common sections using robust detection
    found_sections = find_resume_sections(resume_text)
    core_sections = ["experience", "education", "skills"]
    core_section_count = sum(1 for section in core_sections if section in found_sections)
    
    # Only flag as missing if NONE of the core sections are found
    if core_section_count == 0:
        issues.append("Missing important sections")
        suggestions.append("Ensure your resume has clear Experience, Education, and/or Skills sections")
    
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
        "word_count": word_count,
        "found_sections": found_sections
    }

def calculate_experience_from_dates(text):
    """
    Calculate years of experience from date ranges found in text.
    Handles regular hyphen (-), en-dash (–), and em-dash (—).
    
    Detects patterns like:
    - "Aug 2025 – Present" / "Aug 2025 - Present"
    - "Mar 2025 – May 2025"
    - "2020 - Present" / "2020 – 2023"
    
    Returns:
        tuple: (years, found_dates) where years is float or 0, found_dates is bool
    """
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Separator pattern: matches hyphen (-), en-dash (–), or em-dash (—)
    sep = r'\s*[\-\u2013\u2014]\s*'
    
    # Date range patterns
    patterns = [
        # Month Year – Present (e.g., "Aug 2025 – Present")
        r'([A-Za-z]{3,})\s+(\d{4})' + sep + r'(present|current|now)',
        # Month Year – Month Year (e.g., "Mar 2025 – May 2025")
        r'([A-Za-z]{3,})\s+(\d{4})' + sep + r'([A-Za-z]{3,})\s+(\d{4})',
        # Year – Present (e.g., "2020 – Present")
        r'(\d{4})' + sep + r'(present|current|now)',
        # Year – Year (e.g., "2020 – 2023")
        r'(\d{4})' + sep + r'(\d{4})',
    ]
    
    today = datetime.date.today()
    total_months = 0
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            try:
                # Pattern: Month Year – Present
                if len(groups) == 3 and groups[2].lower() in ['present', 'current', 'now']:
                    month = month_map.get(groups[0].lower(), 1)
                    year = int(groups[1])
                    start = datetime.date(year, month, 1)
                    months = (today.year - start.year) * 12 + (today.month - start.month)
                    total_months += max(months, 0)
                
                # Pattern: Month Year – Month Year
                elif len(groups) == 4:
                    month1 = month_map.get(groups[0].lower(), 1)
                    year1 = int(groups[1])
                    month2 = month_map.get(groups[2].lower(), 1)
                    year2 = int(groups[3])
                    months = (year2 - year1) * 12 + (month2 - month1)
                    total_months += max(months, 0)
                
                # Pattern: Year – Present or Year – Year
                elif len(groups) == 2:
                    if groups[1].lower() in ['present', 'current', 'now']:
                        year = int(groups[0])
                        months = (today.year - year) * 12 + today.month
                        total_months += max(months, 0)
                    else:
                        months = (int(groups[1]) - int(groups[0])) * 12
                        total_months += max(months, 0)
            except (ValueError, IndexError):
                continue
    
    if total_months > 0:
        return round(total_months / 12, 1), True
    else:
        return 0, False


def extract_experience_years(text):
    """
    Extract years of experience mentioned in the text.
    First tries text patterns like "5 years experience", then falls back to date-based calculation.
    """
    # Look for patterns like "X years", "X+ years", "5 yrs experience", etc.
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*year\s+(?:professional|work)',
        r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
    ]
    
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years_found.extend([int(y) for y in matches])
    
    if years_found:
        return max(years_found), False  # Return (years, calculated_from_dates)
    
    # Fallback: try to calculate from date ranges
    years_from_dates, found_dates = calculate_experience_from_dates(text)
    if found_dates:
        return years_from_dates, True
    
    return None, False

def analyze_experience(resume_text, jd_text):
    """
    Compare years of experience required vs provided.
    """
    # Extract required years from JD
    jd_result = extract_experience_years(jd_text)
    jd_years = jd_result[0] if isinstance(jd_result, tuple) else jd_result
    
    # Extract resume years
    resume_result = extract_experience_years(resume_text)
    resume_years = resume_result[0] if isinstance(resume_result, tuple) else resume_result
    resume_calculated = resume_result[1] if isinstance(resume_result, tuple) else False
    
    # Also check for experience mentions
    jd_experience_mentions = len(re.findall(r'experience', jd_text.lower()))
    resume_experience_mentions = len(re.findall(r'experience', resume_text.lower()))
    
    result = {
        "jd_required_years": jd_years,
        "resume_years": resume_years,
        "resume_calculated_from_dates": resume_calculated,
        "meets_requirement": None,
        "analysis": []
    }
    
    if jd_years:
        if resume_years:
            if resume_years >= jd_years:
                result["meets_requirement"] = True
                if resume_calculated:
                    result["analysis"].append(f"You have {resume_years} years of experience (calculated from work history), meeting the {jd_years}+ years requirement ✓")
                else:
                    result["analysis"].append(f"You have {resume_years} years of experience, meeting the {jd_years}+ years requirement ✓")
            else:
                result["meets_requirement"] = False
                if resume_calculated:
                    result["analysis"].append(f"You have {resume_years} years of experience (calculated from work history), but the job requires {jd_years}+ years")
                else:
                    result["analysis"].append(f"You have {resume_years} years of experience, but the job requires {jd_years}+ years")
        else:
            result["analysis"].append("Could not determine years of experience from resume. Clearly list your work history with dates.")
    
    if jd_experience_mentions > 0 and resume_experience_mentions < jd_experience_mentions:
        result["analysis"].append("The job description mentions more experience areas. Ensure all relevant experience is clearly documented.")
    
    return result

def detect_education_in_text(text):
    """
    Detect education levels in text using robust regex patterns.
    Handles whitespace normalization for DOCX files and various degree formats.
    
    Args:
        text: Text to search in (resume or job description)
        
    Returns:
        list: List of tuples (education_level, importance_level)
              e.g., [("bachelor", 3), ("master", 4)]
    """
    # Normalize whitespace to handle DOCX formatting issues
    text_clean = ' '.join(text.lower().split())
    
    # Education detection patterns
    education_patterns = {
        "phd": [r'\bph\.?d\b', r'\bdoctorate\b'],
        "master": [r'\bmaster', r'\bm\.sc\b', r'\bmtech\b', r'\bm\.tech\b', r'\bme\b', r'\bpgd\b', r'\bpostgraduate\b', r'\bpost.graduate\b'],
        "bachelor": [r'\bbachelor', r'\bb\.sc\b', r'\bbca\b', r'\bb\.tech\b', r'\bbtec\b', r'\bb\.e\b', r'\bbe\b', r'\bb\.com\b', r'\bbba\b'],
        "associate": [r'\bassociate\b'],
        "high school": [r'\bhigh school\b', r'\bhsc\b', r'\bssc\b', r'\b10\+2\b', r'\bclass 12\b'],
        "mba": [r'\bmba\b']
    }
    
    # Education level hierarchy
    education_levels = {
        "high school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "mba": 4,
        "phd": 5
    }
    
    found = []
    for edu, patterns in education_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_clean):
                found.append((edu, education_levels[edu]))
                break  # Don't add duplicate education levels
    
    return found

def analyze_education(resume_text, jd_text):
    """
    Check if education requirements are met.
    Uses robust pattern matching to handle various degree formats and whitespace issues.
    """
    # Find required education in JD using robust pattern matching
    jd_education = detect_education_in_text(jd_text)
    
    # Find education in resume using robust pattern matching
    resume_education = detect_education_in_text(resume_text)
    
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

def extract_resume_sections(resume_text):
    """
    Extract resume section headers in various formats.
    Detects:
    - Title case headers: "Experience:", "Education:"
    - ALL CAPS headers: "EXPERIENCE", "WORK HISTORY"
    - Headers with or without colons
    
    Args:
        resume_text: Raw resume text
        
    Returns:
        list: Found section names (lowercased)
    """
    common_sections = [
        "experience", "education", "skills", "summary", "objective",
        "certifications", "projects", "awards", "work history",
        "professional experience", "technical skills", "core competencies",
        "qualifications", "achievements", "employment", "references",
        "professional development", "languages"
    ]
    
    found_sections = []
    
    # Create regex patterns for different formats
    patterns = []
    
    # Pattern 1: Title Case with optional colon (e.g., "Experience:" or "Experience")
    # Matches lines that start with capitalized words
    patterns.append(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:?\s*$')
    
    # Pattern 2: ALL CAPS headers (e.g., "EXPERIENCE" or "WORK HISTORY")
    patterns.append(r'^([A-Z\s]{2,})\s*:?\s*$')
    
    lines = resume_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check against common section patterns
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                section_name = matches[0].strip().rstrip(':').lower()
                
                # Only add if it matches a common section name
                for common in common_sections:
                    if section_name == common or common in section_name or section_name in common:
                        if common not in found_sections:
                            found_sections.append(common)
                        break
    
    return found_sections

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

def calculate_composite_ats_score(cosine_sim, matched_keywords, jd_keywords, format_analysis, resume_text):
    """
    Calculate a composite ATS score using multiple factors.
    
    Formula:
    ats_score = (cosine_similarity * 0.35) + (keyword_match_percent * 0.35) + 
                (format_score * 0.15) + (section_score * 0.15)
    
    Args:
        cosine_sim: Cosine similarity score (0-100)
        matched_keywords: List of matched keywords
        jd_keywords: List of all keywords from job description
        format_analysis: Dictionary with format issues
        resume_text: Raw resume text
    
    Returns:
        float: Composite ATS score (0-100)
    """
    # 1. Cosine Similarity (0-100) - weighted at 35%
    similarity_component = cosine_sim * 0.35
    
    # 2. Keyword Match Percentage (0-100) - weighted at 35%
    keyword_match_percent = (len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 0
    keyword_component = keyword_match_percent * 0.35
    
    # 3. Format Score (0-100) - weighted at 15%
    format_score = 100
    if format_analysis["issues"]:
        # Subtract 20 points for each format issue, minimum 0
        format_score = max(0, 100 - (len(format_analysis["issues"]) * 20))
    format_component = format_score * 0.15
    
    # 4. Section Score (0-100) - weighted at 15%
    required_sections = ["experience", "education", "skills", "summary"]
    resume_lower = resume_text.lower()
    found_sections = sum(1 for section in required_sections if section in resume_lower)
    section_score = (found_sections / len(required_sections)) * 100 if required_sections else 0
    section_component = section_score * 0.15
    
    # Calculate composite score
    composite_score = similarity_component + keyword_component + format_component + section_component
    
    return min(100, max(0, composite_score))  # Ensure score is between 0-100

def get_ai_suggestions(resume_text, jd_text, ats_score, missing_keywords, skills_analysis, api_key):
    """
    Get AI-powered suggestions using Anthropic API.
    
    Args:
        resume_text: Raw resume text
        jd_text: Job description text
        ats_score: Current ATS score
        missing_keywords: List of missing keywords
        skills_analysis: Skills analysis dictionary
        api_key: Anthropic API key
        
    Returns:
        str: AI-generated suggestions or error message
    """
    try:
        client = Anthropic(api_key=api_key)
        
        # Prepare the prompt
        missing_technical_skills = ", ".join(skills_analysis.get('technical', {}).get('missing', [])[:10]) if skills_analysis.get('technical', {}).get('missing') else "None"
        missing_keywords_str = ", ".join(missing_keywords[:10]) if missing_keywords else "None"
        
        prompt = f"""You are an expert ATS resume consultant. Analyze this resume against the job description and provide 5 specific, actionable improvements.

ATS Score: {ats_score:.1f}%
Missing Keywords: {missing_keywords_str}
Missing Technical Skills: {missing_technical_skills}

Resume (excerpt): {resume_text[:1500]}

Job Description: {jd_text[:1500]}

Provide exactly 5 numbered, specific, and practical suggestions. Be direct and focus on improving ATS compatibility and relevance to the job."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    except Exception as e:
        return f"Error getting AI suggestions: {str(e)}"

def get_ai_suggestions_ollama(resume_text, jd_text, ats_score, missing_keywords, skills_analysis):
    """
    Get AI-powered suggestions using local Ollama (completely free, no API key).
    Requires Ollama to be running locally.
    
    Args:
        resume_text: Raw resume text
        jd_text: Job description text
        ats_score: Current ATS score
        missing_keywords: List of missing keywords
        skills_analysis: Skills analysis dictionary
        
    Returns:
        str: AI-generated suggestions or error message
    """
    if ollama is None:
        return "Ollama library not installed. Install with: pip install ollama"
    
    try:
        missing_technical_skills = ", ".join(skills_analysis.get('technical', {}).get('missing', [])[:10]) if skills_analysis.get('technical', {}).get('missing') else "None"
        missing_keywords_str = ", ".join(missing_keywords[:10]) if missing_keywords else "None"
        
        prompt = f"""You are an expert ATS resume consultant.

ATS Score: {ats_score:.1f}%
Missing Keywords: {missing_keywords_str}
Missing Technical Skills: {missing_technical_skills}

Resume (excerpt):
{resume_text[:1500]}

Job Description (excerpt):
{jd_text[:1000]}

Provide 5 numbered, specific, and practical improvement suggestions to improve ATS compatibility and match the job description better."""
        
        response = ollama.chat(
            model='llama3.2',
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return response['message']['content']
    
    except Exception as e:
        return f"❌ Ollama not running. Start with: `ollama serve` in a terminal\nError: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Resume ATS Analyzer", page_icon="📄", layout="wide")

st.title("AI Resume ATS Analyzer")
st.markdown("Upload your resume and paste the job description to get a comprehensive ATS compatibility analysis.")

# Sidebar for API key configuration
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key (for AI suggestions)",
        type="password",
        help="Your Anthropic API key to enable AI-powered suggestions. Get one at https://console.anthropic.com"
    )
    if api_key:
        st.session_state.anthropic_api_key = api_key
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- 📊 Composite ATS Score")
    st.markdown("- 🔧 Skills Analysis")
    st.markdown("- 💼 Experience Matching")
    st.markdown("- 🎓 Education Qualifier")
    st.markdown("- 📋 Format Checker")
    if st.session_state.get('anthropic_api_key'):
        st.markdown("- 🤖 AI Suggestions")

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
        # Check if we need to run analysis or use cached results from session_state
        analysis_key = f"resume_analysis_{hash((uploaded_file.name, job_description, industry))}"
        
        if analysis_key not in st.session_state:
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

                # Calculate raw cosine similarity
                cosine_sim = calculate_similarity(processed_resume, processed_jd)

                # Keyword analysis (supports both single words and multi-word phrases)
                jd_keywords = extract_keywords(processed_jd)
                # For keywords, check if they exist as words or phrases in the resume
                matched_keywords = [
                    kw for kw in jd_keywords 
                    if skill_in_text(kw, processed_resume)  # Use word boundary matching
                ]
                missing_keywords = [
                    kw for kw in jd_keywords 
                    if not skill_in_text(kw, processed_resume)
                ]

                # 2. Resume Format Checker (needed for composite ATS score)
                format_analysis = check_resume_format(resume_text, uploaded_file.type)

                # Calculate composite ATS score
                ats_score = calculate_composite_ats_score(cosine_sim, matched_keywords, jd_keywords, format_analysis, resume_text)

                # ====== NEW FEATURES ======
                
                # 1. Skills Analysis
                skills_analysis = analyze_skills(resume_text, job_description)
                
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
                resume_sections = extract_resume_sections(resume_text)
                industry_advice = get_industry_advice(selected_industry, resume_sections)
                
                # 8. Cover Letter Analysis
                cover_letter_analysis = None
                if cover_letter_text:
                    cover_letter_analysis = analyze_cover_letter(cover_letter_text, resume_text, job_description)

                # Suggestions
                suggestions = generate_suggestions(ats_score, matched_keywords, missing_keywords, processed_resume, processed_jd)

                # Store all results in session_state to persist across reruns
                st.session_state[analysis_key] = {
                    'ats_score': ats_score,
                    'matched_keywords': matched_keywords,
                    'missing_keywords': missing_keywords,
                    'skills_analysis': skills_analysis,
                    'format_analysis': format_analysis,
                    'experience_analysis': experience_analysis,
                    'education_analysis': education_analysis,
                    'detected_industry': detected_industry,
                    'selected_industry': selected_industry,
                    'industry_advice': industry_advice,
                    'cover_letter_analysis': cover_letter_analysis,
                    'suggestions': suggestions,
                    'jd_keywords': jd_keywords,
                    'processed_resume': processed_resume,
                    'processed_jd': processed_jd
                }
        
        # Retrieve results from session_state
        results = st.session_state[analysis_key]
        ats_score = results['ats_score']
        matched_keywords = results['matched_keywords']
        missing_keywords = results['missing_keywords']
        skills_analysis = results['skills_analysis']
        format_analysis = results['format_analysis']
        experience_analysis = results['experience_analysis']
        education_analysis = results['education_analysis']
        detected_industry = results['detected_industry']
        selected_industry = results['selected_industry']
        industry_advice = results['industry_advice']
        cover_letter_analysis = results['cover_letter_analysis']
        suggestions = results['suggestions']
        jd_keywords = results['jd_keywords']
        processed_resume = results['processed_resume']
        processed_jd = results['processed_jd']

        # Display results
        st.success("Analysis Complete!")
        
        # Industry Detection
        if industry == "Auto-Detect":
            st.info(f"📊 Detected Industry: **{detected_industry}**")

        # ATS Score with color coding
        st.subheader("📈 ATS Score")
        
        # Determine color and label based on score
        if ats_score < 40:
            color = "#F44336"  # Red
            label = "Needs Major Work"
        elif ats_score < 70:
            color = "#FF9800"  # Orange
            label = "Needs Improvement"
        elif ats_score < 85:
            color = "#2196F3"  # Blue
            label = "Good Match"
        else:
            color = "#4CAF50"  # Green
            label = "Excellent Match"
        
        # Display score with color
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{ats_score:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{label}</h3>", unsafe_allow_html=True)
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
                if experience_analysis.get("resume_calculated_from_dates"):
                    st.metric("Your Experience", f"{experience_analysis['resume_years']} years (from dates)")
                else:
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

        # ====== AI-Powered Suggestions ======
        if st.session_state.get('anthropic_api_key'):
            st.markdown("---")
            st.subheader("🤖 AI-Powered Suggestions")
            
            with st.spinner("Generating AI suggestions..."):
                ai_suggestions = get_ai_suggestions(
                    resume_text, 
                    job_description, 
                    ats_score, 
                    missing_keywords, 
                    skills_analysis,
                    st.session_state.get('anthropic_api_key')
                )
            
            if "Error" in ai_suggestions:
                st.error(ai_suggestions)
            else:
                st.markdown(ai_suggestions)

        # ====== Local AI Suggestions (Ollama) ======
        else:
            st.markdown("---")
            st.subheader("🤖 AI Suggestions (Local)")
            st.info("💡 Use local AI (free, no API key needed). Make sure Ollama is running.")
            
            if st.button("Generate Suggestions with Local AI", key="ollama_suggestions"):
                with st.spinner("Generating suggestions with local AI..."):
                    ollama_suggestions = get_ai_suggestions_ollama(
                        resume_text,
                        job_description,
                        ats_score,
                        missing_keywords,
                        skills_analysis
                    )
                
                if "❌" in ollama_suggestions or "Error" in ollama_suggestions:
                    st.error(ollama_suggestions)
                else:
                    st.markdown(ollama_suggestions)

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
