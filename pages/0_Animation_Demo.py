import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Data contoh
data = ("Data (2).xlsx")

# Membaca data ke dalam DataFrame
df = pd.DataFrame(data, columns=['data'])

# Membuat objek TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Menghitung TF-IDF untuk dokumen
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Dokumen'])

# Membuat DataFrame untuk representasi TF-IDF
tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Menampilkan aplikasi Streamlit
st.title("Contoh Aplikasi TF-IDF di Streamlit")

# Tampilkan data dokumen asli
st.subheader("Data Dokumen Asli:")
st.write(df)

# Tampilkan data TF-IDF
st.subheader("Hasil Perhitungan TF-IDF:")
st.write(tfidf_df)



st.set_page_config(page_title="Topic", page_icon="📹")
st.markdown("# Topic")
st.sidebar.header("Topic Modelling")
st.write(
    """This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters."""
)

# animation_demo()

# show_code(animation_demo)
