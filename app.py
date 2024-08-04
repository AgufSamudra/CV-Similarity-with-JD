import streamlit as st
import os
from pypdf import PdfReader
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import io
import zipfile
import markdown
import google.generativeai as genai
import matplotlib.pyplot as plt

# metode embedding
from tfidf import tfidf
from gemini import gemini
from bert import bert

nltk.download('punkt')
API_KEY="AIzaSyAjVjTEYhTrCwIL_-Q0rBCV7HAOrqTuLd8"
genai.configure(api_key=API_KEY)

def html_template(markdown):
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Summary</title>
</head>
<body>
    <h1>AI Summary</h1>
    <div>
        {markdown}
    </div>
</body>
</html>
"""
    return html_template

def markdown_to_html(markdown_text):
    return markdown.markdown(markdown_text)


def pdf_to_text(uploaded_file, txt_path):
    try:
        # Membaca file PDF yang diunggah
        reader = PdfReader(uploaded_file)
        # Menyimpan semua teks dari PDF
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Menyimpan teks ke file .txt
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        

# Fungsi untuk ekstraksi teks dari PDF menggunakan pypdf
def pdf_to_text_pypdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()  # Menghapus spasi kosong di awal dan akhir
    except Exception as e:
        print(f"Terjadi kesalahan pada {pdf_path}: {e}")
        return None

# Fungsi untuk mengunggah CV dan job description, serta ekstraksi teksnya
def upload_and_extract_text(uploaded_files, csv_path):
    data = []
    for file in uploaded_files:
        text = pdf_to_text_pypdf(file)
        if text:  # Hanya menyimpan teks yang tidak kosong
            data.append({
                'cv_pelamar': text,
                'path': file.name
            })

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, escapechar='\\')
        print(f"Teks berhasil disimpan ke {csv_path}")
    else:
        print("Tidak ada teks yang diekstrak, file CSV tidak dibuat.")


# Streamlit layout
st.title("CV Similarity with Job Description")

metode_embed = st.selectbox("Pilih Meotde", ["-", "BERT", "Gemini", "TF-IDF", "All Compare"])

if metode_embed == "BERT":
    # Upload CVs and Job Description
    uploaded_cvs = st.file_uploader("Upload CV PDFs", type="pdf", accept_multiple_files=True)
    uploaded_job_desc = st.file_uploader("Upload Job Description PDF", type="pdf")
    
    button_search = st.button("Search", type="primary")
    if button_search:
        
        with st.spinner("Processing Data..."):
            
            cv_texts = upload_and_extract_text(uploaded_cvs, "output_csv/cv_pelamar.csv")
            job_description_text = upload_and_extract_text(uploaded_job_desc, "jd.txt")
            
            bert_metode, result_sum, top_n_indices, cv_df, avarage_similarity, average_bleu_score_bert, average_rouge_2_f_score_bert = bert("output_csv/cv_pelamar.csv", "jd.txt")
            
            st.subheader("AI Summary")
            st.markdown(bert_metode)
            
            st.subheader("Summary Evaluation")
            with st.expander("BERT Result Evaluation"):
                st.text(result_sum)
                
            html_mode = markdown_to_html(bert_metode)
            html_template_ = html_template(html_mode)
            
            html_file_path = "saran_ai.html"
            with open(html_file_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_template_)
            
            # Create a ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.write(html_file_path, os.path.basename(html_file_path))
                for index in top_n_indices:
                    cv_path = cv_df.loc[index, "path"]
                    with open(f"dataset/{cv_path}", "rb") as f:
                        zip_file.writestr(os.path.basename(cv_path), f.read())
            
            zip_buffer.seek(0)
            
            # Create download button for the ZIP file
            st.subheader("Download PDF Kandidat")
            st.download_button(
                label="Download PDF",
                data=zip_buffer,
                file_name="cvs.zip",
                mime='application/zip',
                type="primary"
            )

if metode_embed == "Gemini":
    # Upload CVs and Job Description
    uploaded_cvs = st.file_uploader("Upload CV PDFs", type="pdf", accept_multiple_files=True)
    uploaded_job_desc = st.file_uploader("Upload Job Description PDF", type="pdf")
    
    button_search = st.button("Search", type="primary")
    if button_search:
        
        with st.spinner("Processing Data..."):
            
            cv_texts = upload_and_extract_text(uploaded_cvs, "output_csv/cv_pelamar.csv")
            job_description_text = pdf_to_text(uploaded_job_desc, "jd.txt")
            
            gemini_metode, result_sum, top_n_indices, cv_df, avarage_similarity, average_bleu_score_gemini, average_rouge_2_f_score_gemini = gemini("output_csv/cv_pelamar.csv", "jd.txt")
            st.subheader("AI Summary")
            st.markdown(gemini_metode)
            
            st.subheader("Summary Evaluation")
            with st.expander("Gemini Result Evaluation"):
                st.text(result_sum)
                
            html_mode = markdown_to_html(gemini_metode)
            html_template_ = html_template(html_mode)
            
            html_file_path = "saran_ai.html"
            with open(html_file_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_template_)
            
            # Create a ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.write(html_file_path, os.path.basename(html_file_path))
                for index in top_n_indices:
                    cv_path = cv_df.loc[index, "path"]
                    with open(f"dataset/{cv_path}", "rb") as f:
                        zip_file.writestr(os.path.basename(cv_path), f.read())
            
            zip_buffer.seek(0)
            
            # Create download button for the ZIP file
            st.subheader("Download PDF Kandidat")
            st.download_button(
                label="Download PDF",
                data=zip_buffer,
                file_name="cvs.zip",
                mime='application/zip',
                type="primary"
            )

if metode_embed == "TF-IDF":
    # Upload CVs and Job Description
    uploaded_cvs = st.file_uploader("Upload CV PDFs", type="pdf", accept_multiple_files=True)
    uploaded_job_desc = st.file_uploader("Upload Job Description PDF", type="pdf")
    
    button_search = st.button("Search", type="primary")
    if button_search:
        
        with st.spinner("Processing Data..."):
            
            cv_texts = upload_and_extract_text(uploaded_cvs, "output_csv/cv_pelamar.csv")
            job_description_text = pdf_to_text(uploaded_job_desc, "jd.txt")
            
            
            tfidf_metode, result_sum, top_n_indices, cv_df, avarage_similarity, average_bleu_score_tfidf, average_rouge_2_f_score_tfidf = tfidf("output_csv/cv_pelamar.csv", "jd.txt")
            st.subheader("AI Summary")
            st.markdown(tfidf_metode)
            
            st.subheader("Summary Evaluation")
            with st.expander("TF-IDF Result Evaluation"):
                st.text(result_sum)
                
            html_mode = markdown_to_html(tfidf_metode)
            html_template_ = html_template(html_mode)
            
            html_file_path = "saran_ai.html"
            with open(html_file_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_template_)
                
            # Create a ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.write(html_file_path, os.path.basename(html_file_path))
                for index in top_n_indices:
                    cv_path = cv_df.loc[index, "path"]
                    with open(f"dataset/{cv_path}", "rb") as f:
                        zip_file.writestr(os.path.basename(cv_path), f.read())
            
            zip_buffer.seek(0)
            
            # Create download button for the ZIP file
            st.subheader("Download PDF Kandidat")
            st.download_button(
                label="Download Data",
                data=zip_buffer,
                file_name="cvs.zip",
                mime='application/zip',
                type="primary"
            )
            

if metode_embed == "All Compare":
    
    # Upload CVs and Job Description
    uploaded_cvs = st.file_uploader("Upload CV PDFs", type="pdf", accept_multiple_files=True)
    uploaded_job_desc = st.file_uploader("Upload Job Description PDF", type="pdf")
    
    button_search = st.button("Search", type="primary")
    if button_search:
        
        with st.spinner("Processing Data..."):
            
            cv_texts = upload_and_extract_text(uploaded_cvs, "output_csv/cv_pelamar.csv")
            job_description_text = upload_and_extract_text(uploaded_job_desc, "jd.txt")
            
            bert_metode, bert_result_sum, bert_top_n_indices, bert_cv_df, avarage_similarity_bert, average_bleu_score_bert, average_rouge_2_f_score_bert = bert("output_csv/cv_pelamar.csv", "jd.txt")
            gemini_metode, gemini_result_sum, gemini_top_n_indices, gemini_cv_df, avarage_similarity_gemini, average_bleu_score_gemini, average_rouge_2_f_score_gemini = gemini("output_csv/cv_pelamar.csv", "jd.txt")
            tfidf_metode, tfidf_result_sum, tfidf_top_n_indices, tfidf_cv_df, avarage_similarity_tfidf, average_bleu_score_tfidf, average_rouge_2_f_score_tfidf = tfidf("output_csv/cv_pelamar.csv", "jd.txt")
            
            st.subheader("Evaluation Embedding")
            col1, col2, col3 = st.columns(3)
            
            # Prepare the prompt for the Gemini API
            prompt = f"""
            BERT MATRIX EVALUATION
            {bert_result_sum}
            
            
            GEMINI MATRIX EVALUATION
            {gemini_result_sum}
            
            
            TFIDF MATRIX EVALUATINO
            {tfidf_result_sum}

            =======================================================

            Berdasarkan hasil evaluasi embedding dari ketiga metode di atas, jika dilihat dari Blue Score dan Rouge.
            Metode Embedding yang mana yang menghasilkan kinerja yang paling baik, urutkan dari yang paling bagus ke yang paling jelek.

            """

            # Generate the response with Gemini API
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
            
            with col1:
                with st.expander("BERT Evaluation"):
                    st.text(bert_result_sum)
            
            with col2:
                with st.expander("Gemini Evaluation"):
                    st.text(gemini_result_sum)
            
            with col3:
                with st.expander("TF-IDF Evaluation"):
                    st.text(tfidf_result_sum)
                    
            st.subheader("AI Evaluation")
            st.markdown(response.text)
            
            st.subheader("Plot Mean Similarity")
            # Dataframe
            df = pd.DataFrame({
                "Metode": ["BERT", "Gemini", "TF-IDF"],
                "Similarity": [avarage_similarity_bert, avarage_similarity_gemini, avarage_similarity_tfidf],
                "Bleu Score": [average_bleu_score_bert, average_bleu_score_gemini, average_bleu_score_tfidf],
                "Rouge Score": [average_rouge_2_f_score_bert, average_rouge_2_f_score_gemini, average_rouge_2_f_score_tfidf]
            })

            # Plot Mean Similarity
            plt.figure(figsize=(8, 4))
            plt.bar(df["Metode"], df["Similarity"], color='skyblue')
            plt.xlabel("Metode")
            plt.ylabel("Similarity")
            plt.title("Mean Similarity")
            st.pyplot(plt.gcf())
            plt.clf()  # Clear figure to avoid overlap

            # Plot Mean Bleu
            st.subheader("Plot Mean Bleu")
            plt.figure(figsize=(8, 4))
            plt.bar(df["Metode"], df["Bleu Score"], color='lightgreen')
            plt.xlabel("Metode")
            plt.ylabel("Bleu Score")
            plt.title("Mean Bleu Score")
            st.pyplot(plt.gcf())
            plt.clf()  # Clear figure to avoid overlap

            # Plot Mean Rouge
            st.subheader("Plot Mean Rouge")
            plt.figure(figsize=(8, 4))
            plt.bar(df["Metode"], df["Rouge Score"], color='salmon')
            plt.xlabel("Metode")
            plt.ylabel("Rouge Score")
            plt.title("Mean Rouge Score")
            st.pyplot(plt.gcf())
            plt.clf()  # Clear figure to 
    
    
            
# # Combine and count the most common indices
# combined_indices = indices_sentence_transformer + indices_gemini + indices_tfidf
# frequency = {}
# for index in combined_indices:
#     if index in frequency:
#         frequency[index] += 1
#     else:
#         frequency[index] = 1
# st.write("Most common CVs based on combined rankings:")
# sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
# for number, count in sorted_frequency[:10]:
#     st.write(f"CV Index: {number}, Count: {count}")
