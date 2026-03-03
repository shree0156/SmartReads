# SmartReads – Book Recommendation System
## Overview

SmartReads is a hybrid book recommendation system that suggests personalized books based on user preferences. It combines content-based filtering and collaborative filtering techniques to generate meaningful recommendations.

## Problem Statement

Online platforms often struggle to provide relevant book recommendations tailored to individual user interests. This project aims to build a recommendation system that enhances user experience by suggesting books based on similarity and user behavior.

## Tech Stack

Python

Pandas, NumPy

Scikit-learn

Surprise (SVD for collaborative filtering)

Streamlit

## Approach
### 1. Data Preprocessing

Cleaned and transformed book dataset

Handled missing values

Created feature representations for content similarity

### 2. Content-Based Filtering

Used TF-IDF vectorization

Calculated cosine similarity between books

### 3. Collaborative Filtering

Implemented SVD model using Surprise library

Learned latent user-book interactions

### 4. Recommendation Logic

Combined similarity-based filtering and user rating predictions

Returned top-N personalized recommendations

### 5.Results

Successfully generated personalized book suggestions

Demonstrated effective similarity matching and rating prediction

## How to Run

Clone the repository

Install dependencies

Run the Streamlit app or Python script
