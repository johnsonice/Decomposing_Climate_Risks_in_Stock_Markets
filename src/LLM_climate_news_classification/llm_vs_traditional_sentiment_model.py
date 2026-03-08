#!/usr/bin/env python3
"""Compare LLM-based (Llama-3.1-8B) vs traditional sentiment models (VADER, TextBlob)
on climate news classification.

This script evaluates how well general-purpose sentiment analyzers perform on the
domain-specific task of classifying climate news paragraphs as favorable or unfavorable
to climate-friendly policies, compared to an LLM with few-shot chain-of-thought prompting.

Usage:
    python llm_vs_traditional_sentiment_model.py
    python llm_vs_traditional_sentiment_model.py --output-dir ./results --vader-threshold 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = ['favorable', 'unfavorable']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare LLM (Llama-3.1-8B) vs traditional sentiment models (VADER, TextBlob) '
                    'on climate news classification'
    )
    parser.add_argument(
        '--train-file', type=str,
        default='/data/home/xiong/data/Fund/Climate/training_eval_results/'
                'Llama-3.1-8B-Instruct_long_fewshot_cot_train_results_v2.csv',
        help='Path to the training evaluation results CSV'
    )
    parser.add_argument(
        '--val-file', type=str,
        default='/data/home/xiong/data/Fund/Climate/training_eval_results/'
                'Llama-3.1-8B-Instruct_long_fewshot_cot_val_results_v2.csv',
        help='Path to the validation evaluation results CSV'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/data/home/xiong/data/Fund/Climate/llama_vs_traditional_method',
        help='Directory to save output CSVs and plots'
    )
    parser.add_argument(
        '--vader-threshold', type=float, default=0.0,
        help='VADER compound score threshold: >= threshold -> favorable, < threshold -> unfavorable (default: 0.0)'
    )
    parser.add_argument(
        '--textblob-threshold', type=float, default=0.0,
        help='TextBlob polarity threshold: >= threshold -> favorable, < threshold -> unfavorable (default: 0.0)'
    )
    parser.add_argument(
        '--neutral-strategy', type=str, default='favorable',
        choices=['favorable', 'unfavorable'],
        help='How to map exact-zero (neutral) scores (default: favorable)'
    )
    return parser.parse_args()


def _check_dependencies():
    """Verify that optional dependencies are available."""
    missing = []
    try:
        import nltk  # noqa: F401
    except ImportError:
        missing.append("nltk (pip install nltk)")
    try:
        from textblob import TextBlob  # noqa: F401
    except ImportError:
        missing.append("textblob (pip install textblob)")
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install nltk textblob")
        raise SystemExit(1)


def ensure_nltk_data() -> None:
    """Download VADER lexicon if not already available."""
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate a results CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Drop spurious index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    required_cols = {'paragraph', 'true_label', 'predicted_label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    return df


def predict_vader(texts: pd.Series, threshold: float = 0.0,
                  neutral_strategy: str = 'favorable') -> list:
    """Classify texts using VADER sentiment analyzer.

    Maps compound score: > threshold -> favorable, < threshold -> unfavorable,
    == threshold -> neutral_strategy.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    predictions = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            predictions.append(neutral_strategy)
            continue
        compound = sid.polarity_scores(text)['compound']
        if compound > threshold:
            predictions.append('favorable')
        elif compound < threshold:
            predictions.append('unfavorable')
        else:
            predictions.append(neutral_strategy)

    return predictions


def predict_textblob(texts: pd.Series, threshold: float = 0.0,
                     neutral_strategy: str = 'favorable') -> list:
    """Classify texts using TextBlob sentiment polarity.

    Maps polarity: > threshold -> favorable, < threshold -> unfavorable,
    == threshold -> neutral_strategy.
    """
    from textblob import TextBlob

    predictions = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            predictions.append(neutral_strategy)
            continue
        polarity = TextBlob(text).sentiment.polarity
        if polarity > threshold:
            predictions.append('favorable')
        elif polarity < threshold:
            predictions.append('unfavorable')
        else:
            predictions.append(neutral_strategy)

    return predictions


def clean_llama_predictions(labels: pd.Series,
                            neutral_strategy: str = 'favorable') -> list:
    """Clean Llama predicted labels by mapping neutral -> favorable and handling nulls.

    The Llama-3.1-8B predictions contain 2 'neutral' labels in the training set
    despite the binary setup. This follows the project convention of merging
    neutral with favorable (see merge_pred_results.py).
    """
    mapping = {
        'favorable': 'favorable',
        'unfavorable': 'unfavorable',
        'neutral': neutral_strategy,
    }
    cleaned = []
    for label in labels:
        if pd.isna(label):
            cleaned.append(neutral_strategy)
        else:
            cleaned.append(mapping.get(str(label).strip().lower(), neutral_strategy))
    return cleaned


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute accuracy, precision, recall, and F1 score (macro + per-class)."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    for label in LABEL_NAMES:
        metrics[f'precision_{label}'] = precision_score(
            y_true, y_pred, labels=[label], average='micro', zero_division=0)
        metrics[f'recall_{label}'] = recall_score(
            y_true, y_pred, labels=[label], average='micro', zero_division=0)
        metrics[f'f1_{label}'] = f1_score(
            y_true, y_pred, labels=[label], average='micro', zero_division=0)

    metrics['classification_report'] = classification_report(
        y_true, y_pred, labels=LABEL_NAMES, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)

    return metrics


def build_comparison_table(results_dict: dict) -> pd.DataFrame:
    """Build a tidy comparison table from nested {split: {model: metrics}} results."""
    metric_keys = [
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'precision_favorable', 'recall_favorable', 'f1_favorable',
        'precision_unfavorable', 'recall_unfavorable', 'f1_unfavorable',
    ]
    rows = []
    for split_name, models in results_dict.items():
        for model_name, metrics in models.items():
            row = {'split': split_name, 'model': model_name}
            for key in metric_keys:
                row[key] = metrics.get(key)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['split', 'f1_macro'], ascending=[True, False])
    return df


def create_disagreement_df(df: pd.DataFrame, model_preds: dict) -> pd.DataFrame:
    """Create per-sample DataFrame with all model predictions and agreement info."""
    result = df[['paragraph', 'true_label']].copy()
    if 'justification' in df.columns:
        result['justification'] = df['justification']

    for model_name, preds in model_preds.items():
        result[f'pred_{model_name}'] = preds

    pred_cols = [f'pred_{m}' for m in model_preds.keys()]
    result['all_agree'] = result[pred_cols].nunique(axis=1) == 1

    def get_correct_models(row):
        correct = [m for m in model_preds if row[f'pred_{m}'] == row['true_label']]
        return ', '.join(correct) if correct else 'none'

    result['correct_models'] = result.apply(get_correct_models, axis=1)
    return result


def filter_llm_advantage(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where Llama predicted correctly but both VADER and TextBlob got it wrong.

    These examples illustrate the LLM's contextual understanding advantage
    over traditional keyword-based sentiment models.
    """
    llama_correct = disagreement_df['pred_Llama-3.1-8B'] == disagreement_df['true_label']
    vader_wrong = disagreement_df['pred_VADER'] != disagreement_df['true_label']
    textblob_wrong = disagreement_df['pred_TextBlob'] != disagreement_df['true_label']

    return disagreement_df[llama_correct & vader_wrong & textblob_wrong].copy()


def plot_confusion_matrices(results_dict: dict, split_name: str,
                            output_dir: Path) -> None:
    """Plot side-by-side confusion matrices for all models on a given split."""
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        cm = results_dict[model_name]['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name}\n({split_name} set)', fontsize=12)
        ax.set_xticks(range(len(LABEL_NAMES)))
        ax.set_yticks(range(len(LABEL_NAMES)))
        ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

    fig.suptitle(f'Confusion Matrices — {split_name.capitalize()} Set',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / f'confusion_matrices_{split_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot grouped bar charts comparing metrics across models and splits."""
    macro_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    splits = comparison_df['split'].unique()
    models = comparison_df['model'].unique()
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    # --- Macro metrics ---
    fig, axes = plt.subplots(1, len(splits), figsize=(8 * len(splits), 6), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        split_data = comparison_df[comparison_df['split'] == split]
        x = np.arange(len(macro_metrics))
        width = 0.25

        for i, (_, row) in enumerate(split_data.iterrows()):
            values = [row[m] for m in macro_metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=row['model'],
                          color=colors[i % len(colors)])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{split.capitalize()} Set', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Model Comparison: Macro Metrics', fontsize=14)
    plt.tight_layout()
    out_path = output_dir / 'metrics_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # --- Per-class F1 ---
    fig, axes = plt.subplots(1, len(splits), figsize=(8 * len(splits), 5), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        split_data = comparison_df[comparison_df['split'] == split]
        x = np.arange(len(models))
        width = 0.35
        f1_fav = split_data['f1_favorable'].values
        f1_unfav = split_data['f1_unfavorable'].values
        model_labels = split_data['model'].values

        ax.bar(x - width / 2, f1_fav, width, label='favorable', color='#4CAF50')
        ax.bar(x + width / 2, f1_unfav, width, label='unfavorable', color='#f44336')
        ax.set_title(f'{split.capitalize()} Set', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Class F1 Scores by Model', fontsize=14)
    plt.tight_layout()
    out_path = output_dir / 'f1_per_class.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary(comparison_df: pd.DataFrame, llm_advantage_counts: dict) -> None:
    """Print a formatted comparison summary to stdout."""
    display_cols = ['split', 'model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    display_df = comparison_df[display_cols].copy()
    for col in display_cols[2:]:
        display_df[col] = display_df[col].map(lambda x: f'{x:.4f}')

    print("\n" + "=" * 80)
    print("  MODEL COMPARISON SUMMARY: LLM vs Traditional Sentiment Models")
    print("=" * 80)
    print(display_df.to_string(index=False))
    print("=" * 80)

    print("\n  LLM Advantage Cases (Llama correct, VADER+TextBlob both wrong):")
    for split_name, count in llm_advantage_counts.items():
        print(f"    {split_name}: {count} samples")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies and download VADER lexicon
    _check_dependencies()
    ensure_nltk_data()

    # Load data
    print("Loading data...")
    datasets = {
        'train': load_data(args.train_file),
        'val': load_data(args.val_file),
    }
    for name, df in datasets.items():
        n_fav = (df['true_label'] == 'favorable').sum()
        n_unfav = (df['true_label'] == 'unfavorable').sum()
        print(f"  {name}: {len(df)} samples ({n_fav} favorable, {n_unfav} unfavorable)")

    # Run predictions and evaluation for each split
    all_results = {}
    all_disagreements = {}
    llm_advantage_counts = {}

    for split_name, df in datasets.items():
        print(f"\n--- Processing {split_name} set ({len(df)} samples) ---")
        texts = df['paragraph']
        y_true = df['true_label'].tolist()

        # Llama predictions (already in CSV)
        llama_preds = clean_llama_predictions(
            df['predicted_label'], neutral_strategy=args.neutral_strategy)

        # VADER predictions
        print("  Running VADER...")
        vader_preds = predict_vader(
            texts, threshold=args.vader_threshold,
            neutral_strategy=args.neutral_strategy)

        # TextBlob predictions
        print("  Running TextBlob...")
        textblob_preds = predict_textblob(
            texts, threshold=args.textblob_threshold,
            neutral_strategy=args.neutral_strategy)

        model_preds = {
            'Llama-3.1-8B': llama_preds,
            'VADER': vader_preds,
            'TextBlob': textblob_preds,
        }

        # Compute metrics
        split_results = {}
        for model_name, preds in model_preds.items():
            metrics = compute_metrics(y_true, preds)
            split_results[model_name] = metrics
            print(f"\n  {model_name} Classification Report ({split_name}):")
            print(metrics['classification_report'])

        all_results[split_name] = split_results

        # Build disagreement DataFrame
        disagreement_df = create_disagreement_df(df, model_preds)
        all_disagreements[split_name] = disagreement_df

        # Filter LLM advantage cases
        llm_adv = filter_llm_advantage(disagreement_df)
        llm_advantage_counts[split_name] = len(llm_adv)

        # Save disagreement and LLM advantage CSVs
        dis_path = output_dir / f'disagreements_{split_name}.csv'
        disagreement_df.to_csv(dis_path, index=False)
        n_disagree = (~disagreement_df['all_agree']).sum()
        print(f"  Saved disagreements: {dis_path} "
              f"({n_disagree}/{len(disagreement_df)} with disagreement)")

        adv_path = output_dir / f'llm_advantage_{split_name}.csv'
        llm_adv.to_csv(adv_path, index=False)
        print(f"  Saved LLM advantage cases: {adv_path} ({len(llm_adv)} samples)")

    # Build and save comparison table
    comparison_df = build_comparison_table(all_results)
    comparison_csv = output_dir / 'metrics_comparison.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved metrics comparison: {comparison_csv}")

    # Print summary
    print_summary(comparison_df, llm_advantage_counts)

    # Generate visualizations
    print("\nGenerating visualizations...")
    for split_name in all_results:
        plot_confusion_matrices(all_results[split_name], split_name, output_dir)
    plot_metrics_comparison(comparison_df, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
