import pandas as pd
import ast


#load movie dataset from csv
def load_data(filepath: str) -> pd.DataFrame:
    
    return pd.read_csv(filepath)


#convert string columns into space separated strings of names
def parse_json_column(series: pd.Series) -> pd.Series:
    
    
    def extract_names(x):
        if pd.isna(x):
            return ""
        try:
            data = ast.literal_eval(x)
            return " ".join(item["name"] for item in data)
        except Exception:
            return ""

    return series.apply(extract_names)


#Clean and preprocess movie dataset.
def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    
    # Drop rows with no title (safety check)
    df = df.dropna(subset=["title"])

    # Handle text columns
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")

    # Parse JSON-like columns
    df["genres"] = parse_json_column(df["genres"])
    df["keywords"] = parse_json_column(df["keywords"])

    # Convert release_date to year
    df["release_date"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    )
    df["release_year"] = df["release_date"].dt.year

    # Fill numeric missing values
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["vote_average"] = df["vote_average"].fillna(0)
    df["vote_count"] = df["vote_count"].fillna(0)

    return df.reset_index(drop=True)