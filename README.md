# SmartReads — Hybrid Book Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-orange?style=flat-square)
![Surprise](https://img.shields.io/badge/Surprise-SVD-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=flat-square&logo=streamlit)](https://shree0156-credit-risk-prediction-model-app-nunrbe.streamlit.app/)

> **Find your next destined book** — SmartReads combines content-based and collaborative filtering into a single hybrid engine that solves the cold-start problem and surfaces genuinely personalised recommendations.

---

## The Problem

Most book recommendation systems use either:
- **Content-based filtering only** — recommends books similar to what you searched, but ignores community taste and ratings
- **Collaborative filtering only** — learns from user behaviour, but completely fails for new or low-activity users (the cold-start problem)

**SmartReads solves both** by fusing the two approaches with an equal-weight hybrid scoring formula.

---

## How It Works

```
User Input (title or author)
        │
        ▼
┌───────────────────────┐     ┌──────────────────────────┐
│  Content-Based Filter │     │  Collaborative Filter    │
│  TF-IDF Vectorization │     │  SVD Model (Surprise lib)│
│  Cosine Similarity    │     │  Predicted User Rating   │
└──────────┬────────────┘     └────────────┬─────────────┘
           │                               │
           └──────────────┬────────────────┘
                          ▼
             Hybrid Score = 0.5 × sim_score
                         + 0.5 × pred_rating
                          │
                          ▼
              Top-N Books ranked by final score
                  displayed with cover image
```

### Step-by-step

| Step | What happens |
|------|-------------|
| **1. Input** | User types a book title or author name |
| **2. Fuzzy match** | System searches both `book_title` and `book_author` fields, picks the closest match |
| **3. Content score** | TF-IDF vectors are built from book metadata; cosine similarity returns the top 10 similar books |
| **4. Collaborative score** | SVD model (trained on user-book ratings) predicts a rating for each candidate book |
| **5. Hybrid fusion** | Final score = equal blend of similarity score and predicted rating |
| **6. Output** | Top recommendations displayed with cover image, author, predicted rating, and similarity score |

---

## Sample Output

> Search: **"Harry"**

| # | Book Title | Author | Predicted Rating | Similarity |
|---|-----------|--------|-----------------|------------|
| 1 | Harry Potter and the Chamber of Secrets | J.K. Rowling | 4.21 | 0.94 |
| 2 | Harry Potter and the Prisoner of Azkaban | J.K. Rowling | 4.18 | 0.91 |
| 3 | Harry Potter and the Goblet of Fire | J.K. Rowling | 4.15 | 0.89 |

> Search: **"Agatha Christie"**

| # | Book Title | Author | Predicted Rating | Similarity |
|---|-----------|--------|-----------------|------------|
| 1 | Murder on the Orient Express | Agatha Christie | 4.35 | 0.96 |
| 2 | And Then There Were None | Agatha Christie | 4.29 | 0.93 |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Content-based filtering | TF-IDF Vectorization + Cosine Similarity (Scikit-learn) |
| Collaborative filtering | SVD — Singular Value Decomposition (Surprise library) |
| Hybrid fusion | Custom weighted scoring (equal 50/50 blend) |
| Web application | Streamlit |
| Model serialisation | Joblib (.pkl) |
| Data handling | Pandas, NumPy |

---

## Project Structure

```
SmartReads/
│
├── SmartReads.ipynb              # Full notebook: EDA, model training, evaluation
├── SmartReads.py                 # Streamlit app (main entry point)
│
├── filtered_books_df.pkl         # Preprocessed book dataset
├── cosine_sim.pkl                # Precomputed TF-IDF cosine similarity matrix
├── book_title_indices.pkl        # Title → index mapping for fast lookup
├── svd_model.pkl                 # Trained SVD collaborative filtering model
│
├── Screenshot 2025-05-27.png     # App screenshot
├── Synopsis.pdf                  # Project synopsis document
└── README.md
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/shree0156/SmartReads.git
cd SmartReads
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn scikit-surprise joblib
```

**3. Run the app**
```bash
streamlit run SmartReads.py
```

**4. Open in browser**
```
http://localhost:8501
```

> **Note:** The `.pkl` model files must be present in the same directory as `SmartReads.py` for the app to load correctly.

<img width="1175" height="1013" alt="demo" src="https://github.com/user-attachments/assets/e82591de-7f35-4037-845d-4d266c107388" />

---

## Key Design Decisions

**Why hybrid over single-method?**
Pure collaborative filtering fails when a user or book has few/no ratings (cold-start). Pure content filtering ignores whether other users actually liked the book. The hybrid approach gets the best of both — content similarity keeps recommendations relevant, while SVD-predicted ratings act as a quality filter.

**Why equal 50/50 weighting?**
After testing different weight combinations during development, equal weighting produced the most balanced results — neither over-indexing on niche similarity matches nor pure popularity bias.

**Why SVD?**
SVD (matrix factorisation) learns latent factors that capture hidden patterns in user-book interactions — it goes beyond simple co-occurrence counts and generalises better to unseen user-book pairs.

---

## Future Improvements

- [ ] Add user login to personalise recommendations based on reading history
- [ ] Incorporate book genre tags and descriptions into TF-IDF features
- [ ] Experiment with neural collaborative filtering (NCF) for improved accuracy
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Add explicit feedback loop (thumbs up/down) to retrain the model

---

## Author

**Shreeja Maiya**
MCA in Artificial Intelligence | Aspiring Data Scientist & ML Engineer

---

*If you found this project useful, consider giving it a ⭐ — it helps others discover it too.*
