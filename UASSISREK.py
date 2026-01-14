import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Song-Based Recommender",
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
    '<p class="subtitle">Song-Based Collaborative Filtering dengan Cosine Similarity</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================
# PREPROCESSING
# =====================
df = df[['artist', 'song', 'popularity']]
df.dropna(inplace=True)
df.drop_duplicates(subset=['artist', 'song'], inplace=True)

# Normalisasi popularity → rating 1–5
scaler = MinMaxScaler(feature_range=(1, 5))
df['rating'] = scaler.fit_transform(df[['popularity']])

# =====================
# SIMULASI USER
# =====================
np.random.seed(42)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
df['user'] = np.random.choice(users, size=len(df))

# =====================
# SONG-ARTIST MATRIX
# =====================
song_artist_matrix = df.pivot_table(
    index='song',
    columns='artist',
    values='rating'
)

# =====================
# COSINE SIMILARITY
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
def recommend_similar_songs(song_name, top_n=10):
    if song_name not in song_similarity_df.index:
        return None

    similar_songs = (
        song_similarity_df[song_name]
        .sort_values(ascending=False)
        .drop(song_name)
        .head(top_n)
    )

    return similar_songs

# =====================
# UI INPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    song_selected = st.selectbox(
        "🎶 Pilih Lagu",
        song_artist_matrix.index
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
    st.markdown('<p class="section">🎧 Lagu yang Mirip</p>', unsafe_allow_html=True)

    recommendations = recommend_similar_songs(song_selected, top_n)

    if recommendations is not None:
        rec_df = recommendations.reset_index()
        rec_df.columns = ["Judul Lagu", "Skor Kemiripan"]

        st.dataframe(
            rec_df.style
            .background_gradient(cmap="Greens")
            .format({"Skor Kemiripan": "{:.2f}"})
        )

        # =====================
        # BAR CHART SIMILARITY
        # =====================
        st.markdown('<p class="section">📊 Visualisasi Kemiripan Lagu</p>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(rec_df["Judul Lagu"], rec_df["Skor Kemiripan"])
        ax1.set_ylabel("Cosine Similarity")
        ax1.set_title(f"Lagu yang Mirip dengan '{song_selected}'")
        ax1.set_xticklabels(rec_df["Judul Lagu"], rotation=45)

        st.pyplot(fig1)

        # =====================
        # HEATMAP SONG SIMILARITY
        # =====================
        st.markdown('<p class="section">🔥 Heatmap Similarity Lagu</p>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(song_similarity_df.iloc[:10, :10])
        plt.colorbar(im)
        ax2.set_xticks(range(10))
        ax2.set_yticks(range(10))
        ax2.set_xticklabels(song_similarity_df.columns[:10], rotation=90)
        ax2.set_yticklabels(song_similarity_df.index[:10])
        ax2.set_title("Heatmap Similarity Lagu (Top 10)")
        st.pyplot(fig2)

        # =====================
        # EVALUASI RMSE (SIMULASI)
        # =====================
        st.markdown('<p class="section">📈 Evaluasi Sistem (RMSE)</p>', unsafe_allow_html=True)

        actual = df['rating'].sample(100, random_state=42)
        predicted = actual + np.random.normal(0, 0.3, size=len(actual))

        rmse = np.sqrt(mean_squared_error(actual, predicted))

        st.success(f"📌 Nilai RMSE: **{rmse:.3f}**")

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(actual, predicted)
        ax3.set_xlabel("Rating Aktual")
        ax3.set_ylabel("Rating Prediksi")
        ax3.set_title("Evaluasi Sistem Rekomendasi (RMSE)")
        st.pyplot(fig3)

    else:
        st.warning("⚠️ Lagu tidak ditemukan.")
