# Weather Retrieval Demo

A small demo that reads an Austin weather CSV, builds simple content-based day profiles (numeric measurements + TF‑IDF on event text), finds similar historical days, and reports brief explanations and retrieval metrics.

## Quick summary
- **Purpose:** Show how to find and explain days in a weather dataset that are similar to a given day.  
- **Input:** austin_weather_labeled.csv (CSV placed in the project folder or passed as a CLI argument).  
- **Output:** console printout of three example days (hot, wet, representative), top‑K similar rows for each, a short human comment per example, and demo retrieval metrics (MAP@K, nDCG@K, MRR@K).

## Files
- **Weather.py** — Main demo script (defaults to ./austin_weather_labeled.csv).  
- **austin_weather_labeled.csv** — Your dataset (not included; supply your own).  
- **outputs/** — (optional) any saved outputs you add later.

## Requirements
- Python 3.8+  
- Packages:
  - pandas
  - numpy
  - scikit-learn

Install dependencies:
```
pip install pandas numpy scikit-learn
```

## How to run
From the project folder:

- Use the default data filename:
```
python Weather.py
```

- Or provide a different CSV path:
```
python Weather.py path/to/your_austin_weather_labeled.csv
```

The script prints three automatically chosen examples and their top‑K similar rows, plus a short comment and summary metrics.

## What the script does (step‑by‑step)
1. Loads the CSV and safely coerces numeric columns (handles "-", "%", commas).  
2. Builds TF‑IDF vectors for a text column (Events/Conditions) and standardizes numeric features.  
3. Combines numeric and text vectors and computes cosine similarity between all days.  
4. Heuristically selects three example days:
   - **hot:** a random day from the top 10% by maximum temperature.  
   - **wet:** a random day from the top 10% by precipitation.  
   - **representative:** the day nearest median temperature and median precipitation.  
5. For each example, lists the top‑K most similar historical days and prints a short interpretation and a brief comment explaining selection.  
6. Computes demo retrieval metrics (MAP@K, nDCG@K, MRR@K) using precip>0 as a placeholder relevance signal.

## Notes on metrics and evaluation
- The metrics reported are for demonstration only. By default the script treats any historical day with precipitation > 0 as “relevant.”  
- For realistic evaluation replace that proxy with true per-query ground-truth relevance sets and re-run the evaluation harness.  
- Metrics included:
  - **MAP@K** — mean average precision at cutoff K.  
  - **nDCG@K** — normalized discounted cumulative gain at K.  
  - **MRR@K** — mean reciprocal rank at K.

## Customization points
- Change numeric/text feature choices by editing the candidate lists at the top of the script.  
- Make example selection deterministic by setting RANDOM_SEED in the script or adding a CLI flag.  
- Replace the demo relevance signal with a CSV file mapping queries to relevant item indices to compute real MAP/nDCG/MRR.

## Troubleshooting
- If you see "CSV file not found", verify the file name and working directory or pass the full path as an argument.  
- If quantile/selection fails due to strings in numeric columns, ensure the CSV uses numeric values or let the script coerce them (it already attempts coercion).  
- For large datasets, TF‑IDF and dense combination may be memory intensive; reduce TFIDF_MAX_FEATURES or use sparse pipelines.
