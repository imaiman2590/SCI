import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
from scipy import stats
from spellchecker import SpellChecker
import unidecode
import re
import fuzzywuzzy
from fuzzywuzzy import fuzz
from phonetics import metaphone
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import joblib
from prometheus_client import start_http_server, Summary, Counter

# -----------------------------
# Prometheus Metrics Initialization
# -----------------------------
REQUESTS = Counter('customer_data_requests', 'Total number of requests')
PROCESS_TIME = Summary('data_processing_duration_seconds', 'Time spent processing data')

# Initialize spell checker and Sentence Transformer model
spell = SpellChecker()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up SQLite connection for storing processed data
conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()

# -----------------------------
# Preprocessing Functions
# -----------------------------

def normalize_text(text):
    """Normalize text to handle different characters and case."""
    return unidecode.unidecode(str(text)).strip().lower() if not pd.isna(text) else ''

def standardize_phone(phone):
    """Standardize phone to 10-digit format."""
    return re.sub(r'\D', '', str(phone))[-10:] if not pd.isna(phone) else ''

def preprocess_data(df, name_col='name', email_col='email', phone_col='phone'):
    """Preprocess the data by normalizing the name, email, and phone fields."""
    df[name_col + '_clean'] = df[name_col].fillna('').apply(normalize_text)
    df[email_col + '_clean'] = df[email_col].fillna('').apply(normalize_text)
    df[phone_col + '_clean'] = df[phone_col].apply(standardize_phone)
    df['combined'] = df[name_col + '_clean'] + ' ' + df[email_col + '_clean']
    return df

def detect_columns(df):
    """Automatically detect relevant columns such as name, email, phone, etc."""
    columns = {
        "name": next((col for col in df.columns if 'name' in col.lower()), None),
        "email": next((col for col in df.columns if 'email' in col.lower()), None),
        "phone": next((col for col in df.columns if 'phone' in col.lower()), None),
        "address": next((col for col in df.columns if 'address' in col.lower()), None),
        "dob": next((col for col in df.columns if 'dob' in col.lower()), None),
        "gender": next((col for col in df.columns if 'gender' in col.lower()), None),
    }
    return columns

def load_dataset(file):
    """Load dataset based on file type (CSV, JSON, SQL, XLSX, TXT)."""
    file_extension = os.path.splitext(file.name)[1].lower()
    loaders = {
        '.csv': pd.read_csv,
        '.json': pd.read_json,
        '.xlsx': pd.read_excel,
        '.txt': lambda file: pd.DataFrame(file.read().decode("utf-8").splitlines(), columns=['raw_data']),
    }
    if file_extension in loaders:
        return loaders[file_extension](file)
    else:
        raise ValueError("Unsupported file type!")

def visualize_columns(dfs):
    """Visualize the column names, data types, and row count of each dataset."""
    all_column_info = []
    for idx, df in enumerate(dfs):
        dataset_name = f"Dataset {idx + 1}"
        column_info = pd.DataFrame({
            'Dataset Name': [dataset_name] * len(df.columns),
            'Column Name': df.columns,
            'Row Count': df.notnull().sum(),
            'Data Type': df.dtypes
        })
        all_column_info.append(column_info)
    column_info_df = pd.concat(all_column_info, ignore_index=True)

    fig = px.bar(column_info_df,
                 x='Column Name',
                 y='Row Count',
                 color='Data Type',
                 title="Column Row Counts with Data Types by Dataset",
                 facet_col='Dataset Name')
    st.plotly_chart(fig)

def build_faiss_index(embeddings):
    """Create a FAISS index from the embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) distance
    index.add(embeddings)
    return index

def remove_outliers(df, z_thresh=3):
    """Remove outliers based on z-scores for numerical columns."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    return df[(z_scores < z_thresh).all(axis=1)]

# -----------------------------
# Embedding + FAISS Indexing
# -----------------------------

def generate_embeddings(df, columns, batch_size=500):
    """Generate embeddings for specified columns using batch processing."""
    text_data = df[columns].fillna(' ').agg(' '.join, axis=1)
    embeddings = [model.encode(text_data[i:i + batch_size].tolist(), convert_to_numpy=True)
                  for i in range(0, len(text_data), batch_size)]
    return np.vstack(embeddings)

def search_similar_customers(df, embeddings, index, top_k=5, threshold=0.85):
    """Search for similar customers based on embeddings."""
    matches = []
    distances, indices = index.search(embeddings, top_k)

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, idxs):
            if i != j:
                sim = 1 / (1 + dist)
                if sim >= threshold or df.iloc[i]['phone_clean'] == df.iloc[j]['phone_clean']:
                    matches.append({
                        'customer_id_1': df.index[i],
                        'customer_id_2': df.index[j],
                        'similarity': round(sim, 3),
                        'same_phone': df.iloc[i]['phone_clean'] == df.iloc[j]['phone_clean']
                    })
    return pd.DataFrame(matches).drop_duplicates()

# -----------------------------
# Phonetic Matching and Similarity Signals
# -----------------------------

def phonetic_matching(name1, name2):
    """Compare names using phonetic matching (Metaphone)."""
    return metaphone.doublemetaphone(name1) == metaphone.doublemetaphone(name2)

def fuzzy_string_matching(str1, str2):
    """Fuzzy matching score between two strings using fuzzywuzzy."""
    return fuzz.token_sort_ratio(str1, str2)

def spell_check(text):
    """Correct spelling using the SpellChecker."""
    return spell.correction(text)

# -----------------------------
# Database Connectivity
# -----------------------------

def save_session_data(df):
    """Save processed data into the database."""
    df.to_sql('customer_data', conn, if_exists='replace', index=False)
    st.success("Session data saved to database.")

def load_session_data():
    """Load previously saved session data from the database."""
    return pd.read_sql("SELECT * FROM customer_data", conn)

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.title("Customer Data Processing")

    # Increment the request counter
    REQUESTS.inc()

    # File uploader
    uploaded_files = st.file_uploader("Upload your datasets", accept_multiple_files=True)
    if uploaded_files:
        dfs = [load_dataset(file) for file in uploaded_files]
        st.write(f"Uploaded {len(dfs)} datasets.")

        # Visualize columns
        visualize_columns(dfs)

        # Set similarity threshold slider
        threshold = st.slider("Set similarity threshold", min_value=0.0, max_value=1.0, value=0.85)

        if st.button("Process Data"):
            with PROCESS_TIME.time():
                integrated_df = pd.concat(dfs, ignore_index=True)
                columns = detect_columns(integrated_df)

                # Ensure required columns are present
                if not all([columns["name"], columns["email"], columns["phone"]]):
                    st.error("❌ Missing required columns: name, email, or phone.")
                    return

                # Handle customer_id as index
                if 'customer_id' in integrated_df.columns:
                    integrated_df.set_index('customer_id', inplace=True)
                else:
                    integrated_df.index.name = 'customer_id'

                # Preprocess data
                integrated_df = preprocess_data(integrated_df, columns["name"], columns["email"], columns["phone"])

                # Generate embeddings
                st.info("🔍 Generating embeddings...")
                embeddings = generate_embeddings(integrated_df, ['combined'])

                # Build FAISS index
                st.info("📦 Building FAISS index...")
                faiss_index = build_faiss_index(embeddings)

                # Search for potential duplicates
                st.info("🔎 Searching for duplicates...")
                matches = search_similar_customers(integrated_df, embeddings, faiss_index, top_k=5, threshold=threshold)

                # Display results
                if not matches.empty:
                    st.success("✅ Potential matches found!")
                    st.dataframe(matches)
                    st.download_button("📥 Download Matches as CSV", matches.to_csv(index=False), file_name="faiss_matches.csv")
                else:
                    st.success("🎉 No potential duplicates found!")

                # Save session data
                if st.button("Save Session Data"):
                    save_session_data(integrated_df)

                # Load session data
                if st.button("Load Previous Session Data"):
                    loaded_df = load_session_data()
                    st.write(loaded_df)

if __name__ == "__main__":
    # Start Prometheus server
    start_http_server(8000)
    main()
