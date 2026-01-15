import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎧",
    layout="wide"
)

st.title("🎵 Sistem Rekomendasi Lagu")
st.markdown("**Content-Based Filtering menggunakan Cosine Similarity**")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("songs_normalize.csv")
    features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'popularity'
    ]
    df = df[['artist', 'song'] + features]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['song'], inplace=True)
    return df, features

df, features = load_data()

# ===============================
# PREPROCESSING
# ===============================
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[features])

similarity_matrix = cosine_similarity(feature_matrix)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df['song'],
    columns=df['song']
)

# ===============================
# FUNCTION RECOMMENDATION
# ===============================
def recommend_songs(song_name, top_n=5):
    scores = (
        similarity_df[song_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )
    return scores

# ===============================
# FUNCTION EVALUATION
# ===============================
def precision_at_k(song_name, k=5, threshold=0.7):
    top_k = (
        similarity_df[song_name]
        .drop(song_name)
        .sort_values(ascending=False)
        .head(k)
    )
    relevance = top_k >= threshold
    precision = relevance.sum() / k

    result_df = pd.DataFrame({
        "Song": top_k.index,
        "Similarity": top_k.values,
        "Relevan": relevance.values
    })
    return precision, result_df

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Pengaturan")

song_input = st.sidebar.selectbox(
    "🎶 Pilih Lagu",
    sorted(df['song'].unique())
)

top_n = st.sidebar.slider(
    "Jumlah Rekomendasi (Top-N)",
    min_value=3,
    max_value=10,
    value=5
)

threshold = st.sidebar.slider(
    "Threshold Relevansi",
    min_value=0.5,
    max_value=0.9,
    value=0.7,
    step=0.05
)

# ===============================
# RECOMMENDATION RESULT
# ===============================
st.subheader(f"📌 Lagu Dipilih: **{song_input}**")

recommendations = recommend_songs(song_input, top_n)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎧 Rekomendasi Lagu")
    st.dataframe(
        recommendations.reset_index()
        .rename(columns={"index": "Song", song_input: "Similarity"}),
        use_container_width=True
    )

with col2:
    st.markdown("### 📊 Visualisasi Similarity")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(recommendations.index, recommendations.values)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Top Similar Songs")
    st.pyplot(fig)

# ===============================
# EVALUATION
# ===============================
st.markdown("---")
st.subheader("📈 Evaluasi Sistem (Precision@K)")

precision, eval_df = precision_at_k(song_input, top_n, threshold)

st.metric(
    label=f"Precision@{top_n}",
    value=f"{precision:.2f}"
)

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("### 📋 Detail Evaluasi")
    st.dataframe(eval_df, use_container_width=True)

with col4:
    st.markdown("### 📊 Bar Evaluasi")

    colors = ['green' if r else 'red' for r in eval_df['Relevan']]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(eval_df['Song'], eval_df['Similarity'], color=colors)
    ax2.axvline(x=threshold, linestyle='--')
    ax2.invert_yaxis()
    ax2.set_xlabel("Similarity")
    ax2.set_title("Relevansi Rekomendasi")
    st.pyplot(fig2)

# ===============================
# PIE CHART
# ===============================
st.markdown("### 🥧 Distribusi Relevansi")

relevant_count = eval_df['Relevan'].sum()
non_relevant_count = top_n - relevant_count

fig3, ax3 = plt.subplots(figsize=(4, 4))
ax3.pie(
    [relevant_count, non_relevant_count],
    labels=["Relevan", "Tidak Relevan"],
    autopct="%1.1f%%",
    startangle=90
)
ax3.axis("equal")
st.pyplot(fig3)

st.success("✅ Sistem rekomendasi berhasil dijalankan!")
