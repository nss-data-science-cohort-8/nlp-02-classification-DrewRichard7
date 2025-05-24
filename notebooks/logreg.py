import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# 1. Load & split the data
complaints = pd.read_csv("../data/complaints.csv")
complaints = complaints.rename(
    columns={
        "Consumer complaint narrative": "consumer_complaint_narrative",
        "Issue": "issue",
    }
)
X = complaints["consumer_complaint_narrative"]
y = complaints["issue"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Build a Pipeline with a placeholder 'vect' step
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# 3. Grid over two vectorizers and two n-gram ranges
param_grid = {
    "vect": [
        TfidfVectorizer(max_features=5000, stop_words="english"),
        CountVectorizer(max_features=5000, stop_words="english"),
    ],
    "vect__ngram_range": [(1, 1), (1, 2)],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,  # fewer folds → faster
    scoring="f1_macro",
    n_jobs=-1,  # parallelize across cores
    verbose=2,
)
grid.fit(X_train, y_train)

# 4. Report which vectorizer & n-gram range won
best_vect = grid.best_params_["vect"]
best_ngrams = grid.best_params_["vect__ngram_range"]
print(f"Best vectorizer: {type(best_vect).__name__}")
print(f"Best ngram_range: {best_ngrams}")

# 5. Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Extract most influential words/phrases per class
vect = best_model.named_steps["vect"]
clf = best_model.named_steps["clf"]
feat_names = vect.get_feature_names_out()
coefs = clf.coef_  # shape = (n_classes, n_features)
classes = clf.classes_
topn = 10

for idx, cls in enumerate(classes):
    top_idx = np.argsort(coefs[idx])[::-1][:topn]
    top_features = [feat_names[i] for i in top_idx]
    print(f"\nTop {topn} features for class '{cls}':")
    print(top_features)
