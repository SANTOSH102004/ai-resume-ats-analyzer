# AI Resume ATS Analyzer

A Stream.lit-based web application that analyzes resumes against job descriptions using AI techniques to calculate ATS (Applicant Tracking System) compatibility scores with comprehensive resume analysis and improvement suggestions.

## Features

### Core Features
- **Resume Upload**: Support for PDF and DOCX files
- **Job Description Input**: Text area for pasting job descriptions
- **ATS Score Calculation**: Uses TF-IDF vectorization and cosine similarity
- **Keyword Matching**: Identifies matched and missing keywords
- **Score Breakdown**: Detailed analysis with similarity score and keyword match percentage
- **Suggestions**: Automated improvement suggestions based on analysis
- **Download Report**: Export analysis results as a text file
- **Modern UI**: Clean, professional interface with progress bars and metrics

### Advanced Features
- **Skills Analysis**: Categorizes skills into technical and soft skills, shows matched/missing skills with visual indicators
- **Resume Format Checker**: Analyzes resume length, sections, action verbs usage, quantifiable achievements, and file format compatibility
- **Experience Matching**: Compares years of experience required vs provided, checks experience mentions
- **Education Qualifier**: Checks if degree requirements are met (High School, Associate, Bachelor, Master, PhD)
- **Industry-Specific Templates**: Auto-detects industry and provides tailored advice for:
  - Technology
  - Finance
  - Healthcare
  - Marketing
  - Engineering
- **Cover Letter Analysis (Optional)**: Analyzes cover letter compatibility with resume and job description

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**: scikit-learn, pandas, numpy, nltk, PyPDF2, python-docx, pdfplumber

## Installation

1. Clone or download the project files.
2. Install the required dependencies:

   
```
bash
   pip install -r requirements.txt
   
```

3. Run the application:

   
```
bash
   streamlit run app.py
   
```

4. Open your browser to the provided local URL (usually http://localhost:8501).

## Usage

1. Upload your resume (PDF or DOCX format).
2. (Optional) Upload a cover letter for additional analysis.
3. Paste the job description in the text area.
4. Select the industry (or use Auto-Detect).
5. Click "Analyze Resume" to get the ATS score and detailed analysis.
6. Review all the analysis sections:
   - ATS Score with progress bar
   - Score Breakdown (Similarity, Keyword Match, Improvement Potential, Word Count)
   - Skills Analysis (Technical and Soft skills)
   - Experience Analysis (Years required vs provided)
   - Education Analysis (Degree requirements met)
   - Resume Format Checker
   - Industry-Specific Advice
   - Cover Letter Analysis (if provided)
   - Keyword Analysis (Matched/Missing)
   - Suggestions for Improvement
7. Download the comprehensive analysis report.

## Sample Data

### Sample Job Description

```
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
```

### Sample Resume Text

```
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
```

## Project Structure

- `app.py`: Main Streamlit application with all analysis features
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## How It Works

### ATS Score Calculation
1. Extract text from uploaded resume (PDF/DOCX)
2. Preprocess text (lowercase, remove special characters, stopwords, lemmatization)
3. Convert to TF-IDF vectors
4. Calculate cosine similarity between resume and job description
5. Display as percentage score

### Skills Analysis
- Categorizes skills into Technical (programming languages, frameworks, databases, cloud/devops, data science) and Soft skills
- Compares resume skills against job description requirements
- Shows matched and missing skills with visual indicators

### Experience & Education Matching
- Extracts years of experience from both resume and job description
- Compares education levels required vs provided
- Provides clear indicators if requirements are met

## Contributing

Feel free to fork and contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the MIT License.

---

Made with ❤️ for job seekers and recruiters
