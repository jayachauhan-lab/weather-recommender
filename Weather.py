﻿#!/usr/bin/env python3
# coding: utf-8
"""
SupervisedWeatherModel.py

- Input: austin_weather_labeled.csv
- Task: Train a supervised classifier to predict weather labels ("Hot", "Wet", "Mild", "Cold")
- Features: temp, tempmax, precip
- Model: RandomForestClassifier
- Evaluation: Accuracy, F1-score, Confusion Matrix
- Ranking Metrics: F1@3, MAP@3, MRR
- Example Predictions: Randomly selected from test set with commentary

Usage:
  python SupervisedWeatherModel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------
# Configuration
# ----------------------
CSV_PATH = Path("austin_weather_labeled.csv")
LABEL_COLUMN = "label"
NUMERIC_COLUMNS = ["temp", "tempmax", "precip"]
RANDOM_SEED = 42
TOP_K = 3

# ----------------------
# Load and clean data
# ----------------------
def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].replace("T", 0).replace("-", np.nan)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("Unlabeled")
    df = df[df[LABEL_COLUMN] != "Unlabeled"]
    return df

# ----------------------
# Feature engineering
# ----------------------
def build_features(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[NUMERIC_COLUMNS])
    y = df[LABEL_COLUMN]
    return X, y, scaler

# ----------------------
# Ranking metrics
# ----------------------
def f1_at_k(y_true, y_probs, label_encoder, k=3):
    top_k = np.argsort(y_probs, axis=1)[:, -k:][:, ::-1]
    y_true_idx = label_encoder.transform(y_true)
    hits = [1 if y_true_idx[i] in top_k[i] else 0 for i in range(len(y_true))]
    return sum(hits) / len(hits)

def map_at_k(y_true, y_probs, label_encoder, k=3):
    top_k = np.argsort(y_probs, axis=1)[:, -k:][:, ::-1]
    y_true_idx = label_encoder.transform(y_true)
    scores = []
    for i in range(len(y_true)):
        if y_true_idx[i] in top_k[i]:
            rank = np.where(top_k[i] == y_true_idx[i])[0][0] + 1
            scores.append(1 / rank)
        else:
            scores.append(0)
    return np.mean(scores)

def mrr(y_true, y_probs, label_encoder):
    top_k = np.argsort(y_probs, axis=1)[:, ::-1]
    y_true_idx = label_encoder.transform(y_true)
    scores = []
    for i in range(len(y_true)):
        ranks = np.where(top_k[i] == y_true_idx[i])[0]
        scores.append(1 / (ranks[0] + 1) if len(ranks) > 0 else 0)
    return np.mean(scores)

# ----------------------
# Train and evaluate
# ----------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train_enc)
    y_pred_enc = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    y_probs = model.predict_proba(X_test)
    print("\n=== Ranking Metrics ===")
    print(f"F1@{TOP_K}: {f1_at_k(y_test, y_probs, label_encoder, k=TOP_K):.3f}")
    print(f"MAP@{TOP_K}: {map_at_k(y_test, y_probs, label_encoder, k=TOP_K):.3f}")
    print(f"MRR: {mrr(y_test, y_probs, label_encoder):.3f}")

    return model, label_encoder, X_test, y_test

# ----------------------
# Example predictions with commentary
# ----------------------
def run_examples(model, scaler, label_encoder, X_test, y_test):
    print("\n=== Example Predictions (Randomized) ===")
    df_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=NUMERIC_COLUMNS)
    df_test[LABEL_COLUMN] = y_test.values
    sampled = df_test.sample(n=3, random_state=RANDOM_SEED)
    for i, row in sampled.iterrows():
        X_ex = pd.DataFrame([row[NUMERIC_COLUMNS]], columns=NUMERIC_COLUMNS)
        X_scaled = scaler.transform(X_ex)
        probs = model.predict_proba(X_scaled)[0]
        top_k = np.argsort(probs)[-TOP_K:][::-1]
        labels = label_encoder.inverse_transform(top_k)
        print(f"\nExample {i}:")
        print(f"  Input = {row[NUMERIC_COLUMNS].to_dict()}")
        print(f"  True Label = {row[LABEL_COLUMN]}")
        for rank, label in enumerate(labels, 1):
            print(f"  Rank {rank}: {label} (score = {probs[top_k[rank-1]]:.3f})")
        # Commentary
        if labels[0] == "Hot":
            print("  → Commentary: High tempmax likely triggered 'Hot' classification.")
        elif labels[0] == "Wet":
            print("  → Commentary: Elevated precipitation suggests a 'Wet' day.")
        elif labels[0] == "Cold":
            print("  → Commentary: Low temp and tempmax with no rain match 'Cold' conditions.")
        elif labels[0] == "Mild":
            print("  → Commentary: Moderate temp and low precip suggest a 'Mild' classification.")
        else:
            print("  → Commentary: Prediction does not match expected rules.")

# ----------------------
# Main
# ----------------------
def main():
    df = load_and_prepare(CSV_PATH)
    X, y, scaler = build_features(df)
    model, label_encoder, X_test, y_test = train_and_evaluate(X, y)
    run_examples(model, scaler, label_encoder, X_test, y_test)

if __name__ == "__main__":
    main()
