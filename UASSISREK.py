import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
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
    '<p class="subtitle">Content-Based Filtering dengan Cosine Similarity</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================
# PRE-PROCESSING
# =====================
features = [
    'danceability',
    'energy',
    'valence',
    'tempo',
    'acousticness'
]

df = df[['song', 'artist'] + features]
df.dropna(inplace=True)

# Normalisasi fitur
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# =====================
# SIMILARITY
# =====================
content_similarity = cosine_similarity(df[features])

content_similarity_df = pd.DataFrame(
    content_similarity,
    index=df['song'],
    columns=df['song']
)

# =====================
# FUNCTION REKOMENDASI
# =====================
def recommend_content_based(song_name, top_n=5):
    if song_name not in content_similarity_df.index:
        return None

    similar_songs = (
        content_similarity_df[song_name]
        .sort_values(ascending=False)[1:top_n+1]
    )
    return similar_songs

# =====================
# UI INPUT
# =====================
song_selected = st.selectbox(
    "🎶 Pilih Lagu:",
    df['song'].unique()
)

top_n = st.slider(
    "🎯 Jumlah Rekomendasi",
    1, 10, 5
)

st.divider()

# =====================
# OUTPUT
# =====================
if st.button("✨ Tampilkan Rekomendasi"):
    recommendations = recommend_content_based(song_selected, top_n)

    if recommendations is not None:
        st.subheader("🎧 Lagu yang Mirip Berdasarkan Konten")

        recommendations_df = recommendations.reset_index()
        recommendations_df.columns = ["Judul Lagu", "Skor Kemiripan"]

        st.dataframe(
            recommendations_df.style
            .background_gradient(cmap="Purples")
            .format({"Skor Kemiripan": "{:.2f}"})
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            recommendations_df["Judul Lagu"],
            recommendations_df["Skor Kemiripan"]
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_title("📊 Visualisasi Kemiripan Lagu (Content-Based)")
        ax.invert_yaxis()

        st.pyplot(fig)

    else:
        st.warning("⚠️ Lagu tidak ditemukan.")
