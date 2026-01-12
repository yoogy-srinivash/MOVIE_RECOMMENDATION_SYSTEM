import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    #combining relevant text features into a single column

    content = (
        df["overview"] + " " +
        df["genres"] + " " +
        df["keywords"]
    )

    return content


def compute_similarity_matrix(content: pd.Series):
    
    #converting text content into TF-IDF vectors and computing cosine similarity
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(content)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


def recommend_movies(
    title: str,
    df: pd.DataFrame,
    similarity_matrix,
    top_n: int = 5
) -> pd.DataFrame:

    #recommend top x similar movies for a given title   

    if title not in df["title"].values:
        raise ValueError("Movie title not found in dataset.")

    idx = df[df["title"] == title].index[0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_indices = [
        i for i, score in similarity_scores[1:top_n + 1]
    ]

    return df.loc[top_indices, ["title", "genres", "vote_average"]]