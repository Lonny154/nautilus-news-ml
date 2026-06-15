import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "price_vs_sma_20",
    "news_count",
    "avg_sentiment",
    "min_sentiment",
    "max_sentiment",
    "negative_news_share",
    "positive_news_share",
]

TARGET = "target_green_tomorrow"


def train_walk_forward(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df = df.sort_values("Date")

    split_idx = int(len(df) * 0.8)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    return model, metrics, test.assign(prediction=preds, probability_green=probs)