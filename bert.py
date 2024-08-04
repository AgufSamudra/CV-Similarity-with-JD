import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
import google.generativeai as genai
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Downloading Model...")
def load_models():
    sentence_embed_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
    print("MODEL TRANSFORMER: indo-sentence-bert-base ------------> READY")
    return sentence_embed_model

def bert(path_cv_csv, path_jd_txt):
    # Download NLTK data if not already present
    nltk.download('punkt')

    # Configure Gemini API with your API key
    API_KEY = "AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8"
    genai.configure(api_key=API_KEY)

    # Load CV data from CSV
    cv_df = pd.read_csv(path_cv_csv)
    cvs = cv_df['cv_pelamar'].tolist()

    # Load job requirements from text file
    with open(path_jd_txt, 'r') as file:
        job_requirements = file.read()

    # Initialize Sentence Transformer model for BERT embeddings
    model = load_models()

    # Compute embeddings for CVs and job requirements
    cv_embeddings = model.encode(cvs, convert_to_tensor=True)
    job_embedding = model.encode(job_requirements, convert_to_tensor=True)

    # Calculate cosine similarities
    similarities = util.pytorch_cos_sim(job_embedding, cv_embeddings).cpu().numpy()

    # Get the indices of the top N most similar CVs
    top_n = 3
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]

    # Initialize ROUGE scorer
    rouge = Rouge()

    # Collect results for response generation
    results = []
    results_summary = []
    top_n_similarities = []  # List to store top N similarity scores
    top_n_bleu_scores = []   # List to store top N BLEU scores
    top_n_rouge_scores = []  # List to store top N ROUGE-2 F-scores

    # Evaluate and display the most relevant CVs
    for rank, index in enumerate(top_n_indices, start=1):
        cv_text = cvs[index]
        similarity_score = similarities[0][index]
        top_n_similarities.append(similarity_score)  # Store the similarity score

        # Tokenize texts
        candidate_tokens = nltk.word_tokenize(cv_text)
        reference_tokens = nltk.word_tokenize(job_requirements)
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
        top_n_bleu_scores.append(bleu_score)  # Store the BLEU score
        
        # Calculate ROUGE score
        rouge_scores = rouge.get_scores(cv_text, job_requirements)[0]
        rouge_2_f = rouge_scores['rouge-2']['f']
        top_n_rouge_scores.append(rouge_2_f)  # Store the ROUGE-2 F-score
        
        result_summary = f"Index: {index}\nSimilarity: {similarity_score}\nBLEU Score: {bleu_score}\nROUGE-2 F-Score: {rouge_2_f}"
        
        # Collect results
        result = f"""
    Index: {index}
    Similarity: {similarity_score}
    BLEU Score: {bleu_score}
    ROUGE-2 F-Score: {rouge_2_f}
    CV: {cv_text}

    ========================================================

    JD: {job_requirements}"""
        results.append(result)
        results_summary.append(result_summary)

    # Calculate the average similarity, BLEU, and ROUGE scores
    average_similarity = np.mean(top_n_similarities)
    average_bleu_score = np.mean(top_n_bleu_scores)
    average_rouge_score = np.mean(top_n_rouge_scores)

    # Combine results into a single string
    results_combined = "\n\n".join(results)
    result_sum = "\n\n".join(results_summary)

    # Prepare the prompt for the Gemini API
    prompt = f"""
    {results_combined}

    =======================================================

    Berdasarkan hasil penyaringan CV berikut, kenapa dipilih dan mengapa?
    dan berikan index datanya keberapa.
    """

    # Generate the response with Gemini API
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
    
    return response.text, result_sum, top_n_indices, cv_df, average_similarity, average_bleu_score, average_rouge_score
