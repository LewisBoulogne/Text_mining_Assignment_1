from sklearn.datasets import fetch_20newsgroups
from time import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline

from pathlib import Path



feature_types = {
    "Count Vectorizer": CountVectorizer(),
    "TF Vectorizer": TfidfVectorizer(use_idf=False),
    "TF-IDF Vectorizer": TfidfVectorizer(use_idf=True),
}

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Linear SVC": LinearSVC(max_iter=10000, random_state=42),
    "Ridge Classifier": RidgeClassifier(),
    "Complement NB": ComplementNB(),
}

base_params = {
    "lowercase": True,
    "stop_words": None,
    "analyzer": "word",
    "ngram_range": (1, 1),
    "max_features": None,
}

parameter_values = {
    "lowercase": [True, False],
    "stop_words": [None, "english"],
    "analyzer_ngram": [
        ("word", (1, 1)),
        ("word", (1, 2)),
        ("char", (3, 5)),
    ],
    "max_features": [None, 1000, 5000, 10000, 20000],
}

def plot_results(results, parameter_results):
    # Tasks 2–3: compare feature/classifier combinations
    df = pd.DataFrame(results)

    scores = df.pivot(
        index="Classifier",
        columns="Feature",
        values="F1-Score",
    )

    ax = scores.plot.bar(figsize=(10, 5))
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Classifier and feature comparison")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("plots/classifier_comparison.png", dpi=300)
    plt.close()

    # Task 4: compare settings within each parameter group
    df = pd.DataFrame(parameter_results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (parameter, rows) in zip(axes.flat, df.groupby("Parameter", sort=False)):
        ax.bar(rows["Value"], rows["F1-Score"])
        ax.set_title(parameter)
        ax.set_ylabel("Validation macro F1")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig("plots/parameter_comparison.png", dpi=300)
    plt.close(fig)

def run_parameter_experiments(best_feature, best_classifier, fit_docs, val_docs, y_fit, y_val):
    vectorizer = feature_types[best_feature]
    model = classifiers[best_classifier]
    results = []

    for parameter, values in parameter_values.items():
        for value in values:
            params = base_params.copy()

            if parameter == "analyzer_ngram":
                params["analyzer"], params["ngram_range"] = value
            else:
                params[parameter] = value

            vectorizer.set_params(**params)
            print(f"Testing {parameter}={value}")

            X_train, X_val, train_seconds, val_seconds = vectorize_data(fit_docs, val_docs, vectorizer)

            model, fit_seconds = train_model(model, X_train, y_fit)

            precision, recall, f1, accuracy, predict_seconds = evaluate_model(model, X_val, y_val)

            result = {
                "Feature": best_feature,
                "Classifier": best_classifier,
                "Parameter": parameter,
                "Value": str(value),
                **params,
                "Evaluation split": "validation",
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Accuracy": accuracy,
                "Vectorize train (s)": train_seconds,
                "Vectorize validation (s)": val_seconds,
                "Fit (s)": fit_seconds,
                "Predict (s)": predict_seconds,
            }

            log_result(result, results, filename="results/parameter_results.csv")

    return results

def log_result(result, results, filename="results/results.csv"):
    ''' Logs the result of a single experiment to a CSV file. '''
    results.append(result)
    pd.DataFrame(results).to_csv(filename, index=False)

def load_dataset():
    ''' loads the 20 Newsgroups dataset and returns the training and test data along with their labels. '''
    data_train = fetch_20newsgroups(
        subset="train",
        categories=None, # Task 1: Use all categories
        shuffle=True,
        random_state=42,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        categories=None,
        shuffle=True,
        random_state=42,
    )

    return (
        data_train.data,
        data_test.data,
        data_train.target,
        data_test.target,
        data_train.target_names,
    )

def vectorize_data(train_docs, test_docs, vectorizer):
    ''' applies the given vectorizer to the training and test documents and returns the transformed data along with the time taken for vectorization. '''
    start = time()
    X_train = vectorizer.fit_transform(train_docs)
    train_seconds = time() - start

    start = time()
    X_test = vectorizer.transform(test_docs)
    test_seconds = time() - start

    return X_train, X_test, train_seconds, test_seconds

def train_model(model, X_train, y_train):
    ''' trains the given model on the training data and returns the trained model along with the time taken for training. '''
    start = time()
    model.fit(X_train, y_train)
    fit_seconds = time() - start

    return model, fit_seconds

def evaluate_model(model, X_test, y_test):
    ''' evaluates the given model on the test data and returns precision, recall, f1-score, and the time taken for prediction. '''
    start = time()
    y_pred = model.predict(X_test)
    predict_seconds = time() - start

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy, predict_seconds

def main():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    # train test split
    train_docs, test_docs, y_train, y_test, target_names = load_dataset()
    print(f"Task 1, Number of categories: {len(target_names)}")

    # validation split
    fit_docs, val_docs, y_fit, y_val = train_test_split(
        train_docs,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    results = []
    print(f"Tasks 2 and 3: Comparing classifiers and feature types")

    # selecting feature type
    for feature_name, vectorizer in feature_types.items():
        vectorizer.set_params(**base_params)

        X_train, X_val, train_seconds, val_seconds = vectorize_data(fit_docs, val_docs, vectorizer)

        # selecting classifier
        for classifier_name, model in classifiers.items():
            print(f"Testing {feature_name} with {classifier_name}")

            model, fit_seconds = train_model(model, X_train, y_fit)

            precision, recall, f1, accuracy, predict_seconds = evaluate_model(model, X_val, y_val)

            result = {
                "Feature": feature_name,
                "Classifier": classifier_name,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Accuracy": accuracy,
                "Vectorize train (s)": train_seconds,
                "Vectorize validation (s)": val_seconds,
                "Evaluation split": "validation",
                "Fit (s)": fit_seconds,
                "Predict (s)": predict_seconds,
            }

            log_result(result, results)

    print(pd.DataFrame(results).round(4).to_string(index=False))

    best_result = max(results, key=lambda result: result["F1-Score"])

    best_feature = best_result["Feature"]
    best_classifier = best_result["Classifier"]

    print(f"\nBest feature type: {best_feature}")
    print(f"Best classifier: {best_classifier}")
    print(f"Macro F1: {best_result['F1-Score']:.4f}")

    # task 4: run parameter experiments for the best feature type and classifier 
    parameter_results = run_parameter_experiments(best_feature, best_classifier, fit_docs, val_docs, y_fit, y_val)

    columns = ["Parameter", "Value", "Precision", "Recall", "F1-Score", "Accuracy"]

    print(pd.DataFrame(parameter_results)[columns].round(4).to_string(index=False))

    # Plot results
    plot_results(results, parameter_results)

    best_parameter_result = max(parameter_results,key=lambda result: result["F1-Score"])

    selected_params = {parameter: best_parameter_result[parameter] for parameter in base_params}

    vectorizer = feature_types[best_parameter_result["Feature"]]
    model = classifiers[best_parameter_result["Classifier"]]

    vectorizer.set_params(**selected_params)

    # Refit using all original training docs
    X_train, X_test, train_seconds, test_seconds = vectorize_data(train_docs, test_docs, vectorizer)

    # Refit the selected classifier using all original targets y
    model, fit_seconds = train_model(model, X_train, y_train)

    # Evaluate the fixed configuration on the test split
    precision, recall, f1, accuracy, predict_seconds = evaluate_model(model, X_test, y_test)

    final_result = {
        "Feature": best_parameter_result["Feature"],
        "Classifier": best_parameter_result["Classifier"],
        **selected_params,
        "Evaluation split": "test",
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "Vectorize train (s)": train_seconds,
        "Vectorize test (s)": test_seconds,
        "Fit (s)": fit_seconds,
        "Predict (s)": predict_seconds,
    }

    log_result(final_result, [], filename="results/final_test_results.csv")

    print("\nFinal test results:")
    print(
        pd.DataFrame([final_result])[
            ["Feature", "Classifier", "Precision", "Recall", "F1-Score", "Accuracy"]
        ].round(4).to_string(index=False)
    )

if __name__ == "__main__":
    main()