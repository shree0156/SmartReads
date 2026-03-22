import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import gdown

# --- Download pkl files from Google Drive if not present ---
# Replace each YOUR_XXXX_ID with your actual Google Drive file ID

if not os.path.exists('svd_model_data.pkl'):
    gdown.download('https://drive.google.com/uc?id=1gyT7RDtHK4H-q-Wx6SiIzRFLow67OY92', 'svd_model_data.pkl', quiet=False)

# --- Load all files ---
filtered_books = joblib.load('filtered_books_df.pkl')
cosine_sim     = joblib.load('cosine_sim.pkl')
indices        = joblib.load('book_title_indices.pkl')
model_data     = joblib.load('svd_model_data.pkl')

# --- Unpack SVD components ---
qi           = model_data['qi']
pu           = model_data['pu']
bi           = model_data['bi']
bu           = model_data['bu']
global_mean  = model_data['global_mean']
inner_to_raw = model_data['inner_to_raw']
raw_to_inner = {v: k for k, v in inner_to_raw.items()}

# --- SVD prediction function (no surprise library needed) ---
def svd_predict(isbn):
    iid = str(isbn)
    if iid not in raw_to_inner:
        return float(global_mean)
    inner_id = raw_to_inner[iid]
    pred = global_mean + bi[inner_id] + bu[0] + np.dot(qi[inner_id], pu[0])
    return float(np.clip(pred, 1, 10))

# --- App Config ---
st.set_page_config(page_title="SmartReads - Hybrid Book Recommender", layout="wide")
st.title("SmartReads")
st.caption("**The place where you can find your next destined Book!**")

# --- User Input ---
book_input = st.text_input("Search by book title or author", placeholder="e.g. Harry, Tolkien, Agatha Christie")

# --- Hybrid Recommendation Logic ---
def hybrid_recommendations(user_input):
    user_input = user_input.lower().strip()

    normalized_indices = {k.lower(): v for k, v in indices.items()}

    title_matches  = filtered_books[filtered_books['book_title'].str.lower().str.contains(user_input, na=False)]
    author_matches = filtered_books[filtered_books['book_author'].str.lower().str.contains(user_input, na=False)]
    combined_matches = pd.concat([title_matches, author_matches]).drop_duplicates()

    if combined_matches.empty:
        return None, None

    matched_title = combined_matches.iloc[0]['book_title']
    idx = normalized_indices.get(matched_title.lower())

    if idx is None:
        st.error("Matched book not found in cosine similarity index.")
        return None, None

    # Content-Based Filtering
    sim_scores  = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    sim_indices = [i[0] for i in sim_scores]
    similar_books = filtered_books.iloc[sim_indices].copy()

    # SVD Collaborative Filtering
    similar_books["pred_rating"] = similar_books["isbn"].apply(svd_predict)

    # Similarity scores
    for i, (_, score) in enumerate(sim_scores):
        similar_books.loc[similar_books.index[i], "sim_score"] = score

    # Hybrid score
    similar_books["final_score"] = (
        0.5 * similar_books["sim_score"] +
        0.5 * similar_books["pred_rating"] / 10.0
    )
    return similar_books.sort_values("final_score", ascending=False), matched_title


# --- Display Results ---
if book_input:
    results, matched_title = hybrid_recommendations(book_input)

    if results is None:
        st.warning("No matches found. Try another title or author.")
    else:
        st.success(f"Recommendations based on: **{matched_title}**")
        for _, row in results.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                img_url = row.get("Image-URL-M") or "https://via.placeholder.com/100x150?text=No+Image"
                st.image(img_url, width=100)
            with col2:
                st.subheader(row['book_title'])
                st.caption(f"{row['book_author']} | Genre: {row.get('genre', 'Unknown')}")
                st.write(f"**{row['pred_rating']:.2f}** | Similarity: **{row['sim_score']:.2f}**")
                st.markdown("---")
