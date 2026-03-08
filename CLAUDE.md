# CLAUDE.md — Project Guide for Claude Code

## Project Overview

This repository accompanies the IMF working paper **"Decomposing Climate Risks in Stock Markets"**. It classifies climate-related news using both fine-tuned transformers and LLM-based zero/few-shot approaches, then links the results to equity-market returns.

## Repository Structure

```
.
├── libs/                       # Shared helper utilities (OpenAI, Gemini, Claude, async wrappers)
├── notebooks/                  # Jupyter notebooks for exploration & evaluation
├── src/
│   ├── Fintuning_climate_news_classification/   # HuggingFace fine-tuning pipeline
│   └── LLM_climate_news_classification/         # Prompt engineering & LLM inference scripts
├── docs/                       # Analysis summaries and documentation
├── data/                       # Datasets (not committed to repo)
├── requirements.txt
└── README.md
```

## Key Conventions

- **Python version**: 3.11 (Conda environment `climate-risk`)
- **Dependencies**: managed via `requirements.txt`; install with `pip install -r requirements.txt`
- **API keys**: stored in a `.env` file at project root (loaded via `python-dotenv`); never commit secrets
- **LLM providers used**: OpenAI (GPT-4o), Google (Gemini 1.5), Anthropic (Claude 3), Meta (Llama-3.1-8B via vLLM)
- **Traditional NLP**: NLTK (VADER), TextBlob — used for baseline sentiment comparison

## Important Files

| File | Role |
|------|------|
| `libs/llm_utils.py` | OpenAI helper functions |
| `libs/llm_utils_async.py` | Async LLM inference wrapper |
| `libs/llm_utils_claude.py` | Claude-specific utilities |
| `libs/llm_utils_gemini.py` | Gemini-specific utilities |
| `libs/utils.py` | General utility functions |
| `src/Fintuning_climate_news_classification/train.py` | Fine-tune RoBERTa/BERT classifiers |
| `src/LLM_climate_news_classification/create_batch_task.py` | Split raw news into JSONL batches |
| `src/LLM_climate_news_classification/async_inference.py` | Async, rate-limited LLM inference |
| `src/LLM_climate_news_classification/evaluate_results.py` | Accuracy/F1 metrics & cost aggregation |
| `src/LLM_climate_news_classification/llm_vs_traditional_sentiment_model.py` | LLM vs VADER/TextBlob comparison script |
| `src/LLM_climate_news_classification/llm_vs_traditional_sentiment_model.ipynb` | Notebook version for interactive analysis |
| `docs/analysis_summary.md` | Summary report: LLM vs traditional model comparison |

## Development Notes

- When modifying LLM inference scripts, preserve the existing async/rate-limiting patterns in `libs/llm_utils_async.py`.
- Evaluation metrics follow sklearn conventions (accuracy, precision, recall, F1).
- Batch tasks are stored as JSONL; each line is a self-contained inference request.
- Classification labels: `favorable` / `unfavorable` (binary). Neutral is merged into `favorable` per project convention.
- Results data lives under `/data/home/xiong/data/Fund/Climate/`; comparison outputs go to `llama_vs_traditional_method/` subfolder.
