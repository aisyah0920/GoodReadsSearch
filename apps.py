import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the data
df = pd.read_csv('goodreads_data.csv')

# Select relevant columns
selected_columns = ['Book', 'Author', 'Description', 'Genres', 'URL']
df_selected = df[selected_columns]

# Mockup of the corpus for illustration (replace it with your actual data)
corpus = df_selected['Description'][:100]

# Train a search model (replace this with your actual training process)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Set page configuration
st.set_page_config(
    page_title="Goodreads Semantic Search",
    page_icon="✨",
    layout="centered",
)

# Set custom color styles
st.markdown(
    """
    <style>
        .reportview-container {
            background: #f8f8f8;
        }
        .sidebar .sidebar-content {
            background: #3498db;
            color: #ecf0f1;
        }
        .stButton>button {
            background-color: #2ecc71;
        }
        .stTextInput>div>div>input {
            background-color: #ecf0f1;
            color: #2c3e50;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create the menu
menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    # Build the search interface
    st.title('Goodreads Semantic Search')

    # Add a choice for search parameter
    search_param = st.radio("Search by", ["Book", "Author", "Genre"])

    query = st.text_input('Enter your query')

    if query:
        # Retrieve the most relevant documents
        query_embedding = model.encode([query])[0]

        if search_param == "Book":
            corpus_embeddings = model.encode(corpus.tolist())
        elif search_param == "Author":
            corpus_embeddings = model.encode(df_selected['Author'][:100].tolist())
        elif search_param == "Genre":
            corpus_embeddings = model.encode(df_selected['Genres'][:100].tolist())

        distances = [np.linalg.norm(query_embedding - document_embedding) for document_embedding in corpus_embeddings]

        # Display the results
        sorted_indexes = np.argsort(distances)
        top_documents = sorted_indexes[:10]

        if not top_documents.any():
            st.write("No matching documents found.")
        else:
            for i in top_documents:
                panjangnm = len(df)
                for nomor in range(1, panjangnm):
                    print(nomor)
                st.write(f"Book: {df_selected.iloc[i]['Book']}")
                st.write(f"Author: {df_selected.iloc[i]['Author']}")
                st.write(f"Description: {df_selected.iloc[i]['Description']}")
                st.write(f"Genres: {df_selected.iloc[i]['Genres']}")
                st.write(f"URL: {df_selected.iloc[i]['URL']}")
                st.write("\n")

                # Add author and genre information
                # Contoh loop nomor dari 1 sampai 10
                author_info = df[df['Book'] == df_selected.iloc[i]['Book']]['Author'].values
                genre_info = df[df['Book'] == df_selected.iloc[i]['Book']]['Genres'].values
                st.write(f"Additional Author Info: {author_info}")
                st.write(f"Additional Genre Info: {genre_info}")
                st.write("\n")
                st.write("======================================================================")

elif choice == "About":
    st.title('About Goodreads Semantic Search')
    st.header('GoodReads Platform?')
    st.markdown('<p style="color:green;">Goodreads adalah situs web katalog sosial Amerika dan anak perusahaan Amazon yang memungkinkan individu mencari database buku, anotasi, kutipan, dan ulasannya. Pengguna dapat mendaftar dan mendaftarkan buku untuk menghasilkan katalog perpustakaan dan daftar bacaan. Mereka juga dapat membuat grup sendiri yang berisi saran buku, survei, jajak pendapat, blog, dan diskusi. Kantor situs web berlokasi di San Francisco.</p>', unsafe_allow_html=True)
    st.header('Apa itu Semantic Search?')
    st.markdown('<p style="color:green;">Semantic search adalah sebuah teknologi pencarian yang menggunakan pendekatan berdasarkan makna dan konteks untuk menghasilkan hasil pencarian yang relevan.</p>', unsafe_allow_html=True)
    st.header('Tujuan Semantic Search')
    st.markdown('<p style="color:green;">Memahami maksud sebenarnya dari pencarian pengguna dan memberikan hasil yang lebih tepat dan informatif.</p>', unsafe_allow_html=True)