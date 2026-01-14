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
    return pd.read_csv("songs_normalize.csv")

df = load_data()

# =====================
# STYLE
# =====================
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎵 Sistem Rekomendasi Lagu</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Song-Based Collaborative Filtering dengan Cosine Similarity</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================
# PRE-PROCESSING
# =====================
df = df[['artist', 'song', 'popularity']]
df.dropna(inplace=True)

song_artist_matrix = df.pivot_table(
    index='song',
    columns='artist',
    values='popularity'
)

# =====================
# SIMILARITY
# =====================
song_similarity = cosine_similarity(
    song_artist_matrix.fillna(0)
)

song_similarity_df = pd.DataFrame(
    song_similarity,
    index=song_artist_matrix.index,
    columns=song_artist_matrix.index
)

# =====================
# FUNCTION REKOMENDASI
# =====================
def recommend_similar_songs(song_name, top_n=5):
    if song_name not in song_similarity_df.index:
        return None

    similar_songs = song_similarity_df[song_name].sort_values(ascending=False)[1:top_n+1]
    return similar_songs

# =====================
# UI INPUT
# =====================
song_selected = st.selectbox(
    "🎶 Pilih Lagu:",
    song_artist_matrix.index
)

top_n = st.slider("🎯 Jumlah Rekomendasi", 1, 10, 5)

st.divider()

# =====================
# OUTPUT
# =====================
if st.button("✨ Tampilkan Rekomendasi"):
    recommendations = recommend_similar_songs(song_selected, top_n)

    if recommendations is not None:
        st.subheader("🎧 Lagu yang Mirip")

        recommendations_df = recommendations.reset_index()
        recommendations_df.columns = ["Judul Lagu", "Skor Kemiripan"]

        st.dataframe(
            recommendations_df.style
            .background_gradient(cmap="Greens")
            .format({"Skor Kemiripan": "{:.2f}"})
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            recommendations_df["Judul Lagu"],
            recommendations_df["Skor Kemiripan"]
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_title("📊 Visualisasi Kemiripan Lagu")
        ax.invert_yaxis()

        st.pyplot(fig)
