import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# read files
train_df = pd.read_csv('wire_levels.csv')
predict_df = pd.read_csv('wire.csv')

# clean description
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['clean_description'] = train_df['description'].apply(clean_text)
predict_df['clean_description'] = predict_df['description'].apply(clean_text)

def train_text_classifier(X, y):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

level_models = {}
levels = ['level_1', 'level_2', 'level_3', 'level_4']

for level in levels:
    print(f"Training classifier for {level}...")
    X = train_df['clean_description']
    y = train_df[level]
    model = train_text_classifier(X, y)
    level_models[level] = model

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

for level in levels:
    predict_df[level] = level_models[level].predict(predict_df['clean_description'])

# write output to file
predict_df[['description'] + levels].to_csv('wire_categorized.csv', index=False)
