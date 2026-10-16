# SMS Spam and Ham Classification

Notebook experiments comparing text representations for identifying spam and legitimate (ham) SMS messages.

## Approaches

- `Spam_Ham_Classification_using_NLTK.ipynb`: cleans text, removes stopwords, applies Porter stemming, and compares bag-of-words and TF-IDF features with Multinomial Naive Bayes.
- `Spam_Ham_Classification_using_W2VandAVGW2V.ipynb`: explores pretrained vectors, trains Word2Vec on the SMS corpus, averages token vectors, and trains a Random Forest classifier.
- `SMSSpamCollection.txt`: included tab-separated dataset with `label` and `message` fields.

Both notebooks use train/test splits and display accuracy and classification reports.

## Run locally

Use Python 3 in a virtual environment:

```bash
git clone https://github.com/Sanskar1314/Spam_Ham_Classification.git
cd Spam_Ham_Classification
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas nltk scikit-learn gensim tqdm
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate`. Launch Jupyter from the repository root and run the notebook cells in order.

Before running the text-processing cells, download NLTK resources:

```python
import nltk
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(resource)
```

## Reproduction notes

The Word2Vec notebook starts by downloading `word2vec-google-news-300`, which needs internet access and substantial memory; the later custom Word2Vec experiment trains on the included SMS data. The NLTK notebook has standalone `X` inspection cells without a preceding definition; skip those cells in a fresh run. Some classification-report calls reverse true and predicted labels. Use `classification_report(y_test, y_pred)` when interpreting precision and recall.

The custom Word2Vec representation is trained before the train/test split. For an evaluation without test-corpus influence, fit it on training messages only. No web application or exported classifier is included.
