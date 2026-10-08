from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

from data_setup import load_dataset

if __name__ == "__main__":
    # 1. Define Feature Extractors
    feature_types = {
        'counts': CountVectorizer(),
        'tf': TfidfVectorizer(use_idf=False),
        'tf-idf': TfidfVectorizer(use_idf=True)
    }

    # 2. Define Classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Multinomial NB': MultinomialNB(),
        'Linear SVC': LinearSVC(),
        'Ridge_Classifier': RidgeClassifier(),
        'Complement NB': ComplementNB()
    }

    # 3. Test Combinations
    results = []
    for feat_name, vectorizer in feature_types.items():
        X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(dataset=fetch_20newsgroups, vectorizer=vectorizer)

        
        for clf_name, clf in classifiers.items():
            print(f"Testing {feat_name} vectorizer on {clf_name} classifier")
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            # Calculate macro metrics
            report = classification_report(y_test, preds, output_dict=True)
            results.append({
                'Feature': feat_name,
                'Classifier': clf_name,
                'Precision': report['macro avg']['precision'],
                'Recall': report['macro avg']['recall'],
                'F1-Score': report['macro avg']['f1-score']
            })
            print(results[-1])

    df_results = pd.DataFrame(results)
    print(df_results)