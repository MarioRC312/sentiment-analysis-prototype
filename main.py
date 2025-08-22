import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Cargar dataset
df = pd.read_csv("dataset.csv")

#80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["sentimiento"], test_size=0.2, random_state=42
)

############################################Naive Bayes simple############################################

modelo_nb_simple = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

modelo_nb_simple.fit(X_train, y_train)
pred_nb_simple = modelo_nb_simple.predict(X_test)
print("Naive Bayes simple - Precisión:", accuracy_score(y_test, pred_nb_simple))

for real, pred in zip(y_test, pred_nb_simple):
    print(f"Real: {real} -> Predicho: {pred}")

print("\n" + "-"*40 + "\n")

############################################Naive Bayes con n-grams (1 y 2 palabras)############################################

modelo_nb_ngrams = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("nb", MultinomialNB())
])

modelo_nb_ngrams.fit(X_train, y_train)
pred_nb_ngrams = modelo_nb_ngrams.predict(X_test)
print("Naive Bayes con n-grams - Precisión:", accuracy_score(y_test, pred_nb_ngrams))

for real, pred in zip(y_test, pred_nb_ngrams):
    print(f"Real: {real} -> Predicho: {pred}")

print("\n" + "-"*40 + "\n")

############################################Logistic Regression con n-grams############################################

modelo_lr = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("lr", LogisticRegression(max_iter=1000))
])

modelo_lr.fit(X_train, y_train)
pred_lr = modelo_lr.predict(X_test)
print("Logistic Regression con n-grams - Precisión:", accuracy_score(y_test, pred_lr))

for real, pred in zip(y_test, pred_lr):
    print(f"Real: {real} -> Predicho: {pred}")