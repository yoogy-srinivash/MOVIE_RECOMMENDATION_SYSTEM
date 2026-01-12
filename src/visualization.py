import matplotlib.pyplot as plt
import pandas as pd


def plot_release_year_distribution(df: pd.DataFrame):
    
    #Plot distribution of movies by release year.
    
    plt.figure(figsize=(10, 5))
    df["release_year"].dropna().hist(bins=30)
    plt.title("Distribution of Movie Release Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    plt.show()


def plot_popularity_vs_rating(df: pd.DataFrame):

    #Plot relationship between popularity and vote average.
    
    plt.figure(figsize=(8, 5))
    plt.scatter(df["popularity"], df["vote_average"], alpha=0.5)
    plt.title("Popularity vs Vote Average")
    plt.xlabel("Popularity")
    plt.ylabel("Vote Average")
    plt.show()


def plot_runtime_distribution(df: pd.DataFrame):

    #Plot distribution of movie runtimes.

    plt.figure(figsize=(8, 5))
    df["runtime"].hist(bins=30)
    plt.title("Distribution of Movie Runtime")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Count")
    plt.show()


def plot_top_genres(df: pd.DataFrame, top_n: int = 10):
    
    #plot top N most frequent genres

    #split genre strings into individual genres
    genre_series = df["genres"].str.split().explode()

    genre_counts = genre_series.value_counts().head(top_n)

    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind="bar")
    plt.title(f"Top {top_n} Genres")
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.show()