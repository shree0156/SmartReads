import streamlit as st
import pandas as pd
import joblib
import difflib

from numpy.distutils.misc_util import colour_text

# --- Load Data and Models ---
filtered_books = joblib.load('filtered_books_df.pkl')
cosine_sim = joblib.load('cosine_sim.pkl')
indices = joblib.load('book_title_indices.pkl')
svd_model = joblib.load('svd_model.pkl')

# --- App Config ---
st.set_page_config(page_title="SmartReads - Hybrid Book Recommender", layout="wide")
st.title("SmartReads")
st.caption("**The place where you can find your next destined Book!**")

# --- User Input ---
book_input = st.text_input("Search by book title or author", placeholder="e.g. Harry, Tolkien, Agatha Christie")

# --- Hybrid Recommendation Logic ---
def hybrid_recommendations(user_input):
    user_input = user_input.lower().strip()

    # Create a lowercase version of the index mapping
    normalized_indices = {k.lower(): v for k, v in indices.items()}

    # Match by title or author
    title_matches = filtered_books[filtered_books['book_title'].str.lower().str.contains(user_input)]
    author_matches = filtered_books[filtered_books['book_author'].str.lower().str.contains(user_input)]
    combined_matches = pd.concat([title_matches, author_matches]).drop_duplicates()

    if combined_matches.empty:
        return None, None


    # Take the first matching book for recommendation
    matched_title = combined_matches.iloc[0]['book_title']
    idx = normalized_indices.get(matched_title.lower())

    if idx is None:
        st.error("Matched book not found in cosine similarity index.")
        return None, None

    # Content-Based Filtering
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    sim_indices = [i[0] for i in sim_scores]
    similar_books = filtered_books.iloc[sim_indices].copy()

    # Collaborative Filtering
    similar_books["pred_rating"] = similar_books["isbn"].apply(
        lambda x: svd_model.predict("9999", x).est
    )

    # Combine hybrid score
    for i, (_, score) in enumerate(sim_scores):
        similar_books.loc[similar_books.index[i], "sim_score"] = score

    similar_books["final_score"] = 0.5 * similar_books["sim_score"] + 0.5 * similar_books["pred_rating"]
    hybrid_sorted = similar_books.sort_values("final_score", ascending=False)

    return hybrid_sorted, matched_title


    # --- Content-Based Filtering ---
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    sim_indices = [i[0] for i in sim_scores]
    similar_books = filtered_books.iloc[sim_indices].copy()

    # --- Collaborative Filtering (SVD) ---
    similar_books["pred_rating"] = similar_books["isbn"].apply(
        lambda x: svd_model.predict("9999", x).est
    )

    # --- Combine scores ---
    for i, (_, score) in enumerate(sim_scores):
        similar_books.loc[similar_books.index[i], "sim_score"] = score

    similar_books["final_score"] = (
        0.5 * similar_books["sim_score"] + 0.5 * similar_books["pred_rating"]
    )
    hybrid_sorted = similar_books.sort_values("final_score", ascending=False)

    return hybrid_sorted, matched_title


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
                st.write(f"**{row['pred_rating']:.2f}** |Similarity: **{row['sim_score']:.2f}**")
                st.markdown("---")

