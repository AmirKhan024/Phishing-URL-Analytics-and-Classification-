# Phishing URL Analytics and Classification

Big Data Analytics mini project (topic 66): classify URLs as phishing or
legitimate using lexical and domain-based features.

- **Name:** YOUR NAME
- **Roll number:** YOUR ROLL NUMBER
- **College:** YOUR COLLEGE

## What it does

1. Reads a dataset of about 235,000 labelled URLs.
2. Converts every URL into 37 numeric features, using only the URL text
   (no page download, no WHOIS lookup).
3. Trains and compares four machine-learning models.
4. Shows the analytics and results in an interactive dashboard, where you
   can also check any URL yourself.

## Architecture

    Dataset (CSV of URLs + labels)
            |
    Data processing      features.py   URL -> 37 features
            |
    Machine learning     train.py      4 models, evaluation, best model saved
            |
    Dashboard            app.py        Streamlit + Plotly

## Features

**Lexical (22):** URL length, counts of dots, hyphens, underscores, slashes,
question marks, equals signs, @ symbols, ampersands, percent signs and
digits, digit ratio, letter ratio, special-character count, URL entropy,
path length, path depth, query length, number of query parameters, number
of suspicious keywords (login, verify, secure, ...), double slash in the
path, and longest word length.

**Domain-based (15):** HTTPS used, host length, domain length, number of
subdomains, IP address used instead of a domain, non-standard port, TLD
length, frequently abused TLD, common TLD, digits in the host, hyphens in
the host, host entropy, URL shortener, punycode, and a brand name appearing
outside the main domain.

## Models

Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
(scikit-learn). The data is split 80% for training and 20% for testing.
The best model is chosen by F1 score. Accuracy, precision, recall, F1 and
ROC-AUC are reported for each model.

## Setup

Requires Python 3.10 or newer.

    pip install -r requirements.txt

Download the PhiUSIIL Phishing URL Dataset from
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
then unzip it and put the CSV file inside the `data/` folder.

## Run

    python train.py
    streamlit run app.py

`train.py` takes a few minutes on the full dataset. For a quick run use
`python train.py --sample 50000`. If the `streamlit` command is not found,
use `python -m streamlit run app.py`.

## Project structure

    features.py        feature extraction (lexical + domain-based)
    train.py           training, evaluation and analytics
    app.py             Streamlit dashboard
    requirements.txt   Python libraries
    data/              dataset CSV (not uploaded to GitHub)
    models/            created by train.py (not uploaded to GitHub)

## Dashboard pages

- **Overview:** dataset size, class balance, top-level-domain analytics
- **Feature Analytics:** how each feature differs between the two classes
- **Model Performance:** comparison table, confusion matrix, ROC curves,
  feature importance
- **URL Checker:** classify one URL and list its warning signs
- **Batch Check:** classify many URLs and download the results

## Results

Run `python train.py` and copy the table it prints into this section.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

## Limitations

- The model sees only the URL text. It cannot know what is on the page, how
  old the domain is, or whether the site is on a blocklist.
- In this dataset most legitimate URLs are plain homepages, while phishing
  URLs often have long paths. Test scores are therefore very high, but the
  model can wrongly flag legitimate deep links (long paths, many query
  parameters) as phishing.
- Attackers change their tricks over time, so a model trained on old URLs
  slowly becomes less accurate.

## Future scope

- Add WHOIS-based features such as domain age and registrar.
- Add page-content features (forms, external links, title match).
- Process URLs with PySpark for larger datasets, or score a live stream of
  URLs with Kafka.
- Retrain regularly on fresh phishing feeds.

## Dataset credit

Prasad, A. and Chandra, S. (2024). PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning. Computers & Security. Dataset from the UCI Machine
Learning Repository, licensed CC BY 4.0.
