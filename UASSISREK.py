import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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
    df = pd.read_csv("songs_normalize.csv")
    return df

df = load_data()

# =====================
# STYLE
# =====================
st.markdown("""
<style>
.big-title { font-size: 40px; font-weight: bold; text-align: center; }
.subtitle { text-align: center; color: gray; font-size: 18px; }
.section { font-size: 24px; font-weight: bold; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎵 Sistem Rekomendasi Lagu</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Content-Based Filtering (Cosine Similarity)</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================
# PREPROCESSING
# =====================
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'popularity'
]

df = df[['artist', 'song'] + FEATURES]
df.dropna(inplace=True)
df.drop_duplicates(subset='song', inplace=True)

# =====================
# FEATURE SCALING & SIMILARITY
# =====================
@st.cache_data
def compute_similarity(data):
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(data[FEATURES])
    similarity = cosine_similarity(feature_matrix)

    return pd.DataFrame(
        similarity,
        index=data['song'],
        columns=data['song']
    )

similarity_df = compute_similarity(df)

# =====================
# RECOMMENDATION FUNCTION
# =====================
def recommend_songs(song_name, top_n):
    if song_name not in similarity_df.index:
        return None

    return (
        similarity_df[song_name]
        .drop(song_name)
        .sort_values(ascending=False)
        .head(top_n)
    )

# =====================
# USER INPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    selected_song = st.selectbox("🎶 Pilih Lagu", df['song'].unique())

with col2:
    top_n = st.slider("🎯 Jumlah Rekomendasi", 1, 15, 10)

# =====================
# OUTPUT
# =====================
if st.button("✨ Tampilkan Rekomendasi"):
    results = recommend_songs(selected_song, top_n)

    if results is not None:
        st.markdown('<p class="section">🎧 Lagu Rekomendasi</p>', unsafe_allow_html=True)

        rec_df = results.reset_index()
        rec_df.columns = ['Judul Lagu', 'Skor Kemiripan']

        st.dataframe(
            rec_df.style
            .background_gradient(cmap="Blues")
            .format({"Skor Kemiripan": "{:.3f}"})
        )

        # =====================
        # BAR CHART
        # =====================
        st.markdown('<p class="section">📊 Visualisasi Similarity</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.barh(rec_df['Judul Lagu'], rec_df['Skor Kemiripan'])
        ax.invert_yaxis()
        ax.set_xlabel("Cosine Similarity")
        ax.set_title(f"Lagu Mirip dengan '{selected_song}'")

        st.pyplot(fig)

        # =====================
        # HEATMAP (SELECTED SONG)
        # =====================
        st.markdown('<p class="section">🔥 Heatmap Similarity</p>', unsafe_allow_html=True)

        top_songs = rec_df['Judul Lagu'].tolist() + [selected_song]
        heatmap_data = similarity_df.loc[top_songs, top_songs]

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(heatmap_data)
        plt.colorbar(im)

        ax2.set_xticks(range(len(top_songs)))
        ax2.set_yticks(range(len(top_songs)))
        ax2.set_xticklabels(top_songs, rotation=90)
        ax2.set_yticklabels(top_songs)

        ax2.set_title("Heatmap Similarity Lagu Terpilih")
        st.pyplot(fig2)

    else:
        st.warning("⚠️ Lagu tidak ditemukan.")
