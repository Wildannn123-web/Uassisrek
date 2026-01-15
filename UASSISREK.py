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
st.markdown("""
Sistem rekomendasi lagu berbasis **Content-Based Filtering**
menggunakan **Cosine Similarity** pada fitur audio Spotify.
""")

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
# STATISTIK DATASET
# ===============================
st.markdown("## 📊 Statistik Dataset")

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Lagu", len(df))
c2.metric("Jumlah Artis", df['artist'].nunique())
c3.metric("Rata-rata Popularity", round(df['popularity'].mean(), 2))

st.markdown("---")

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
# FUNCTION
# ===============================
def recommend_songs(song_name, top_n):
    return (
        similarity_df[song_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

def precision_at_k(song_name, k, threshold):
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

def explain_similarity(song_a, song_b):
    diff = abs(
        df[df['song'] == song_a][features].values -
        df[df['song'] == song_b][features].values
    )
    closest = pd.Series(diff[0], index=features).sort_values().head(3)
    return closest.index.tolist()

def radar_chart(song_a, song_b):
    values_a = df[df['song'] == song_a][features].values.flatten()
    values_b = df[df['song'] == song_b][features].values.flatten()

    labels = features + [features[0]]
    values_a = np.append(values_a, values_a[0])
    values_b = np.append(values_b, values_b[0])

    angles = np.linspace(0, 2*np.pi, len(labels))

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values_a, label=song_a)
    ax.plot(angles, values_b, label=song_b)
    ax.fill(angles, values_a, alpha=0.1)
    ax.fill(angles, values_b, alpha=0.1)
    ax.legend(fontsize=8)
    return fig

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
    3, 10, 5
)

threshold = st.sidebar.slider(
    "Threshold Relevansi",
    0.5, 0.9, 0.7, 0.05
)

presentation_mode = st.sidebar.checkbox("🎤 Mode Presentasi")

# ===============================
# PROFIL LAGU
# ===============================
st.markdown(f"## 🎼 Profil Lagu: **{song_input}**")

song_profile = df[df['song'] == song_input][features].T
song_profile.columns = ["Nilai Fitur"]

st.dataframe(song_profile, use_container_width=True)

# ===============================
# REKOMENDASI
# ===============================
st.markdown("## 🎧 Rekomendasi Lagu")

recommendations = recommend_songs(song_input, top_n)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        recommendations.reset_index()
        .rename(columns={"index": "Song", song_input: "Similarity"}),
        use_container_width=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(recommendations.index, recommendations.values)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Top Similar Songs")
    st.pyplot(fig)

# ===============================
# ALASAN REKOMENDASI
# ===============================
st.markdown("## 🔍 Alasan Rekomendasi")

for song in recommendations.index:
    reasons = explain_similarity(song_input, song)
    st.write(f"**{song}** → mirip pada fitur: `{', '.join(reasons)}`")

# ===============================
# EVALUASI
# ===============================
st.markdown("## 📈 Evaluasi Sistem")

precision, eval_df = precision_at_k(song_input, top_n, threshold)

st.metric(f"Precision@{top_n}", f"{precision:.2f}")

col3, col4 = st.columns(2)

with col3:
    st.dataframe(eval_df, use_container_width=True)

with col4:
    colors = ['green' if r else 'red' for r in eval_df['Relevan']]
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.barh(eval_df['Song'], eval_df['Similarity'], color=colors)
    ax2.axvline(x=threshold, linestyle='--')
    ax2.invert_yaxis()
    ax2.set_xlabel("Similarity")
    ax2.set_title("Relevansi Rekomendasi")
    st.pyplot(fig2)

# ===============================
# PIE CHART
# ===============================
st.markdown("## 🥧 Distribusi Relevansi")

relevant = eval_df['Relevan'].sum()
non_relevant = top_n - relevant

fig3, ax3 = plt.subplots(figsize=(4,4))
ax3.pie(
    [relevant, non_relevant],
    labels=["Relevan", "Tidak Relevan"],
    autopct="%1.1f%%",
    startangle=90
)
ax3.axis("equal")
st.pyplot(fig3)

# ===============================
# KESIMPULAN
# ===============================
st.markdown("## 📝 Kesimpulan")

st.write(f"""
Sistem rekomendasi lagu berbasis **Content-Based Filtering**
menggunakan **Cosine Similarity** berhasil memberikan rekomendasi
lagu yang relevan dengan nilai **Precision@{top_n} = {precision:.2f}**.

Kesamaan antar lagu dihitung berdasarkan fitur audio Spotify,
sehingga rekomendasi bersifat personal dan tidak bergantung
pada interaksi pengguna lain.
""")



