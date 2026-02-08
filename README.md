# 📝 Sentiment Analyzer

A clean, portfolio-ready Python project that classifies text as **Positive** or **Negative** using a fine-tuned **DistilBERT** model from [Hugging Face Transformers](https://huggingface.co/transformers/).

---

## 🚀 Features
- 🔍 Sentiment analysis (Positive / Negative)
- 😊 Emoji feedback
- 📊 Confidence bar visualization
- 🧠 Lazy model loading with local cache
- 🧪 Test suite with mocks
- 🧾 CLI + Streamlit UI

---

## 🧱 Project Structure

```
.
├── cli.py
├── sentiment.py
├── utils.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── tests/
	├── test_sentiment.py
	└── test_utils.py
```

---

## 🛠️ Installation

```bash
git clone https://github.com/laila-kz/sentiment-text-analyzer.git
cd sentiment-analyzer

python -m venv .venv
# On Linux/Mac:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate

pip install -r requirements.txt
```

---

## ✅ CLI Usage

Analyze a short string:

```bash
python cli.py --text "I love this project" --output sentiment_result.json
```

Interactive mode (multi-line input until `exit`):

```bash
python cli.py --interactive
```

Optional confidence threshold warning:

```bash
python cli.py --text "Not sure" --threshold 0.7
```

---

## 🌐 Streamlit Usage

```bash
streamlit run sentiment.py
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📌 Notes
- First run may download the model into ./model_cache.
- For best results, ensure you have a stable internet connection on first run.


