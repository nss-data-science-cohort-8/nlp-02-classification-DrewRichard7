# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

## **train a text classification model to identify the issue type based on the consumer complaint narrative.**
complaints = pd.read_csv("../data/complaints.csv")
complaints = complaints.rename(
    columns={
        "Consumer complaint narrative": "consumer_complaint_narrative",
        "Issue": "issue",
    }
)

print(f"\n---Head---\n {complaints.head()}\n")
print(f"\n---Tail---\n {complaints.tail()}\n")
print(f"\n---Describe---\n {complaints.describe()}\n")
print(f"---Shape--- \n{complaints.shape}\n")
print(f"\n---NaNs--- \n{complaints.isna().sum()}\n")
print(f"\n---Issue Type value_counts--- \n{complaints['issue'].value_counts()}")

predictor = "consumer_complaint_narrative"
target = "issue"

X = complaints[predictor]
y = complaints[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words="english",
            ),
        ),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\n----CLASSIFICATION REPORT----\n")
print(classification_report(y_test, y_pred))
print("\n----CONFUSION MATRIX----\n")
print(confusion_matrix(y_test, y_pred))
