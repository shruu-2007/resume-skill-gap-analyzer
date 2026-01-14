import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # remove stopwords
    return " ".join(words)

# Streamlit title
st.title("AI Resume Skill Gap Analyzer 💼")

# Upload resume and job description text files
resume_file = st.file_uploader("Upload your Resume (.txt)", type=["txt"])
job_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

if resume_file and job_file:
    resume_text = resume_file.read().decode("utf-8")
    job_text = job_file.read().decode("utf-8")

    # Clean texts
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    # Create set of resume words
    resume_words = set(resume_clean.split())

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    match_percentage = round(similarity[0][0] * 100, 2)

    # Skills list
    skills_list = ["python", "sql", "machine learning", "deep learning",
                   "data analysis", "pandas", "numpy", "nlp", "data visualization",
                   "ai"]

    # Display results
    st.subheader("Resume Match Percentage")
    st.write(f"{match_percentage} %")

    st.subheader("Skills Overview")
    for skill in skills_list:
        skill_words = skill.split()
        if all(word in resume_words for word in skill_words):
            st.markdown(f"<span style='color:green'>✔ {skill}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'>✖ {skill}</span>", unsafe_allow_html=True)
