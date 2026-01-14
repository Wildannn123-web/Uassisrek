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
st.markdown('<p class="subtitle">Collaborative Filtering (Artist-Based)</p>', unsafe_allow_html=True)

st.divider()

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
# FUNCTION REKOMENDASI (TIDAK DIUBAH)
# =====================
def recommend_songs_by_artist(artist_name, top_n=5):
    if artist_name not in artist_similarity_df.index:
        return None

    similar_artists = artist_similarity_df[artist_name].sort_values(ascending=False)[1:6]
    similar_songs = artist_song_matrix.loc[similar_artists.index]
    mean_scores = similar_songs.mean().sort_values(ascending=False)

    return mean_scores.head(top_n)

# =====================
# UI INPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    artist_selected = st.selectbox(
        "🎤 Pilih Nama Artis:",
        artist_song_matrix.index
    )

with col2:
    top_n = st.slider(
        "🎯 Jumlah Rekomendasi",
        1, 10, 5
    )

st.divider()

# =====================
# OUTPUT
# =====================
if st.button("✨ Tampilkan Rekomendasi"):
    recommendations = recommend_songs_by_artist(artist_selected, top_n)

    if recommendations is not None:
        st.subheader("🎧 Lagu Rekomendasi")

        # =====================
        # PERBAIKAN TABEL (HILANGKAN KOLOM 0)
        # =====================
        recommendations_df = recommendations.reset_index()
        recommendations_df.columns = ["Judul Lagu", "Skor Rekomendasi"]

        st.dataframe(
            recommendations_df.style
            .background_gradient(cmap="Blues")
            .format({"Skor Rekomendasi": "{:.2f}"})
        )

        st.divider()

        # =====================
        # VISUALISASI
        # =====================
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            recommendations_df["Judul Lagu"],
            recommendations_df["Skor Rekomendasi"]
        )
        ax.set_xlabel("Skor Prediksi")
        ax.set_ylabel("Judul Lagu")
        ax.set_title("📊 Visualisasi Rekomendasi Lagu")
        ax.invert_yaxis()

        st.pyplot(fig)

    else:
        st.warning("⚠️ Artis tidak ditemukan dalam dataset.")
