from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

from data_setup import load_dataset

CLASSIFIER = LinearSVC()
CONFIG_SPACE = {'lowercase': [True, False],
                'stop_words': [None, 'english'],
                'analyzer_ngram': [('word', (1, 1)), ('word', (1, 2)), ('char', (3, 5))],
                'max_features': [1000, 5000, 10000, 20000]}

def run_parameter_group_experiment(param_name, param_configs, base_params, results_list):
    """Calculates learning curves for all values of a parameter and plots them together on a single graph."""
    print(f"\n--- Running Group Experiment for: {param_name} ---")

    # Colors and line styles for differentiating parameters on the plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']

    plt.figure(figsize=(9, 5))

    for idx, (param_val, config_override) in enumerate(param_configs):
        params = base_params.copy()
        params.update(config_override)

        vectorizer = TfidfVectorizer(**params)
        X_train, X_test, y_train, y_test, _, _ = load_dataset(
            dataset=fetch_20newsgroups, vectorizer=vectorizer, verbose=False
        )

        train_sizes, train_scores, test_scores = learning_curve(
            CLASSIFIER,
            X_train,
            y_train,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        CLASSIFIER.fit(X_train, y_train)
        preds = CLASSIFIER.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)

        results_list.append({
            'Parameter': param_name,
            'Value': str(param_val),
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        })

        # 4. Plot validation (CV) curve for this parameter setting
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        color = colors[idx % len(colors)]
        style = line_styles[idx % len(line_styles)]

        # Plot cross-validation curve
        plt.plot(
            train_sizes,
            test_mean,
            style,
            color=color,
            linewidth=2,
            label=f'{param_val} (CV F1)'
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.1,
            color=color
        )

    plt.title(f'Learning Curves Comparison: {param_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('Macro F1-Score')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(f'figures/Experiment_{param_name}_learning_curves.png', dpi=300)

if __name__ == '__main__':
    base_params = {
        'lowercase': True,
        'stop_words': None,
        'analyzer': 'word',
        'ngram_range': (1, 1),
        'max_features': None
    }

    results = []

    # a. Lowercasing exploration
    lowercase_configs = [
        (val, {'lowercase': val}) for val in CONFIG_SPACE['lowercase']
    ]
    run_parameter_group_experiment('lowercase', lowercase_configs, base_params, results)

    # b. Stop words exploration
    stopwords_configs = [
        (val, {'stop_words': val}) for val in CONFIG_SPACE['stop_words']
    ]
    run_parameter_group_experiment('stop_words', stopwords_configs, base_params, results)

    # c. Analyzer & N-gram range exploration
    analyzer_ngram_configs = [
        (f"{ana}_{ng}",
         {'analyzer': ana, 'ngram_range': ng})
        for ana, ng in CONFIG_SPACE['analyzer_ngram']
    ]
    run_parameter_group_experiment('analyzer_ngram', analyzer_ngram_configs, base_params, results)

    # d. Max features exploration
    max_features_configs = [
        (val, {'max_features': val}) for val in CONFIG_SPACE['max_features']
    ]
    run_parameter_group_experiment('max_features', max_features_configs, base_params, results)

    df_results = pd.DataFrame(results)
    print("\n================ Task 4 Results Summary ================")
    print(df_results.to_string(index=False))