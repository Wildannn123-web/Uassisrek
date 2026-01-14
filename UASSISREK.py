import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Content-Based Music Recommender",
    layout="wide"
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("songs_normalize.csv")

df = load_data()

# =====================
# STYLE
# =====================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
.section {
    font-size: 26px;
    font-weight: bold;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎵 Sistem Rekomendasi Lagu</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Content-Based Filtering menggunakan Cosine Similarity</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================
# PREPROCESSING
# =====================
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'popularity'
]

df = df[['artist', 'song'] + features]
df.dropna(inplace=True)
df.drop_duplicates(subset=['song'], inplace=True)

# Normalisasi fitur
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[features])

# =====================
# COSINE SIMILARITY
# =====================
similarity_matrix = cosine_similarity(feature_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df['song'],
    columns=df['song']
)

# =====================
# FUNCTION REKOMENDASI CBF
# =====================
def recommend_songs(song_name, top_n=10):
    if song_name not in similarity_df.index:
        return None

    similarity_scores = (
        similarity_df[song_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return similarity_scores

# =====================
# UI INPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    selected_song = st.selectbox(
        "🎶 Pilih Lagu",
        df['song'].unique()
    )

with col2:
    top_n = st.slider(
        "🎯 Jumlah Rekomendasi",
        1, 15, 10
    )

# =====================
# OUTPUT
# =====================
if st.button("✨ Tampilkan Rekomendasi"):
    recommendations = recommend_songs(selected_song, top_n)

    if recommendations is not None:
        st.markdown('<p class="section">🎧 Lagu yang Direkomendasikan</p>', unsafe_allow_html=True)

        rec_df = recommendations.reset_index()
        rec_df.columns = ['Judul Lagu', 'Skor Kemiripan']

        st.dataframe(
            rec_df.style
            .background_gradient(cmap="Blues")
            .format({"Skor Kemiripan": "{:.3f}"})
        )

        # =====================
        # BAR CHART
        # =====================
        st.markdown('<p class="section">📊 Visualisasi Kemiripan Lagu</p>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.barh(
            rec_df['Judul Lagu'],
            rec_df['Skor Kemiripan']
        )
        ax1.set_xlabel("Cosine Similarity")
        ax1.set_title(f"Lagu Mirip dengan '{selected_song}'")
        ax1.invert_yaxis()

        st.pyplot(fig1)

        # =====================
        # HEATMAP (TOP 10)
        # =====================
        st.markdown('<p class="section">🔥 Heatmap Similarity Lagu</p>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(similarity_df.iloc[:10, :10])
        plt.colorbar(im)
        ax2.set_xticks(range(10))
        ax2.set_yticks(range(10))
        ax2.set_xticklabels(similarity_df.columns[:10], rotation=90)
        ax2.set_yticklabels(similarity_df.index[:10])
        ax2.set_title("Heatmap Similarity Lagu (Top 10)")

        st.pyplot(fig2)

    else:
        st.warning("⚠️ Lagu tidak ditemukan.")
