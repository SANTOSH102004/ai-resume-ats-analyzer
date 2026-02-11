# AI Resume ATS Analyzer

A Streamlit-based web application that analyzes resumes against job descriptions using AI techniques to calculate ATS (Applicant Tracking System) compatibility scores.

## Features

- **Resume Upload**: Support for PDF and DOCX files
- **Job Description Input**: Text area for pasting job descriptions
- **ATS Score Calculation**: Uses TF-IDF vectorization and cosine similarity
- **Keyword Matching**: Identifies matched and missing keywords
- **Score Breakdown**: Detailed analysis with similarity score and keyword match percentage
- **Suggestions**: Automated improvement suggestions based on analysis
- **Download Report**: Export analysis results as a text file
- **Modern UI**: Clean, professional interface with progress bars and metrics

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**: scikit-learn, pandas, numpy, nltk, PyPDF2, python-docx, pdfplumber

## Installation

1. Clone or download the project files.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

4. Open your browser to the provided local URL (usually http://localhost:8501).

## Usage

1. Upload your resume (PDF or DOCX format).
2. Paste the job description in the text area.
3. Click "Analyze Resume" to get the ATS score and detailed analysis.
4. Review the score, keyword matches, and suggestions.
5. Optionally download the analysis report.

## Sample Data

### Sample Job Description

```
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

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Contributing

Feel free to fork and contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the MIT License.
