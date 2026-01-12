import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def prepare_features(df: pd.DataFrame):

    #Select features and target for supervised learning.
    
    features = df[
        ["runtime", "vote_count", "popularity", "release_year"]
    ].copy()

    target = df["vote_average"]

    # Fill remaining missing values
    features = features.fillna(0)

    return features, target


def train_model(X, y):

    #Train a regression model.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


#Evaluate the trained model.
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "MSE": mse,
        "R2": r2
    }
