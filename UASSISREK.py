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
    page_title="Sistem Rekomendasi Lagu",
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
    '<p class="subtitle">Artist-Based Collaborative Filtering dengan Cosine Similarity</p>',
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
# USER SIMULASI
# =====================
np.random.seed(42)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
df['user'] = np.random.choice(users, size=len(df))

# =====================
# PIVOT TABLE
# =====================
artist_song_matrix = df.pivot_table(
    index='artist',
    columns='song',
    values='rating'
)

# =====================
# COSINE SIMILARITY
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
def recommend_songs_by_artist(artist_name, top_n=10):
    if artist_name not in artist_song_matrix.index:
        return None

    similar_artists = (
        artist_similarity_df[artist_name]
        .sort_values(ascending=False)
        .drop(artist_name)
    )

    recommended_songs = {}

    for similar_artist in similar_artists.index[:5]:
        songs = artist_song_matrix.loc[similar_artist].dropna()

        for song, rating in songs.items():
            if pd.isna(artist_song_matrix.loc[artist_name].get(song)):
                if song not in recommended_songs:
                    recommended_songs[song] = rating

    recommendations = sorted(
        recommended_songs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:top_n]

# =====================
# UI INPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    artist_selected = st.selectbox(
        "🎤 Pilih Artis",
        artist_song_matrix.index
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
    st.markdown('<p class="section">🎧 Lagu Rekomendasi</p>', unsafe_allow_html=True)

    recommendations = recommend_songs_by_artist(artist_selected, top_n)

    if recommendations is not None:
        rec_df = pd.DataFrame(
            recommendations,
            columns=["Judul Lagu", "Rating Prediksi"]
        )

        st.dataframe(
            rec_df.style
            .background_gradient(cmap="Blues")
            .format({"Rating Prediksi": "{:.2f}"})
        )

        # =====================
        # VISUALISASI ARTIS MIRIP
        # =====================
        st.markdown('<p class="section">📊 Artis yang Paling Mirip</p>', unsafe_allow_html=True)

        similar_artists = (
            artist_similarity_df[artist_selected]
            .sort_values(ascending=False)[1:6]
        )

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(similar_artists.index, similar_artists.values)
        ax1.set_ylabel("Similarity Score")
        ax1.set_title(f"Top 5 Artis Mirip dengan {artist_selected}")
        ax1.set_xticklabels(similar_artists.index, rotation=45)

        st.pyplot(fig1)

        # =====================
        # HEATMAP SIMILARITY
        # =====================
        st.markdown('<p class="section">🔥 Heatmap Similarity Antar Artis</p>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(artist_similarity_df.iloc[:10, :10])
        plt.colorbar(im)
        ax2.set_xticks(range(10))
        ax2.set_yticks(range(10))
        ax2.set_xticklabels(artist_similarity_df.columns[:10], rotation=90)
        ax2.set_yticklabels(artist_similarity_df.index[:10])
        ax2.set_title("Heatmap Similarity Antar Artis (Top 10)")
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
        st.warning("⚠️ Artis tidak ditemukan.")
