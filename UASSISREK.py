import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("songs_normalize.csv")
    return df

df = load_data()

st.title("🎵 Sistem Rekomendasi Lagu Berbasis Collaborative Filtering")

# =====================
# PRE-PROCESSING
# =====================
df = df[['artist', 'song', 'popularity']]
df.dropna(inplace=True)

artist_song_matrix = df.pivot_table(
    index='artist',
    columns='song',
    values='popularity'
)

# =====================
# SIMILARITY
# =====================
artist_similarity = cosine_similarity(
    artist_song_matrix.fillna(0)
)

artist_similarity_df = pd.DataFrame(
    artist_similarity,
    index=artist_song_matrix.index,
    columns=artist_song_matrix.index
)

# =====================
# FUNCTION REKOMENDASI
# =====================
def recommend_songs_by_artist(artist_name, top_n=5):
    if artist_name not in artist_similarity_df.index:
        return None

    similar_artists = artist_similarity_df[artist_name].sort_values(ascending=False)[1:6]
    similar_songs = artist_song_matrix.loc[similar_artists.index]
    mean_scores = similar_songs.mean().sort_values(ascending=False)

    return mean_scores.head(top_n)

# =====================
# UI STREAMLIT
# =====================
artist_selected = st.selectbox(
    "Pilih Nama Artis:",
    artist_song_matrix.index
)

top_n = st.slider("Jumlah Rekomendasi", 1, 10, 5)

if st.button("Tampilkan Rekomendasi"):
    recommendations = recommend_songs_by_artist(artist_selected, top_n)

    if recommendations is not None:
        st.subheader("🎧 Lagu Rekomendasi")
        st.dataframe(recommendations)

        # =====================
        # VISUALISASI
        # =====================
        fig, ax = plt.subplots()
        recommendations.plot(kind='barh', ax=ax)
        ax.set_xlabel("Skor Prediksi")
        ax.set_ylabel("Judul Lagu")
        ax.set_title("Hasil Rekomendasi Lagu")
        st.pyplot(fig)
    else:
        st.warning("Artis tidak ditemukan dalam dataset.")
