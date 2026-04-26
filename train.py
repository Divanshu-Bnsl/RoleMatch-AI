import pandas as pd
import pickle
import re
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

os.makedirs("model", exist_ok=True)

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("data/resume.csv")
df = df[['Resume_str', 'Category']]
df.rename(columns={'Resume_str': 'Resume'}, inplace=True)

# ---------------------------
# 2. CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned'] = df['Resume'].apply(clean_text)

# ---------------------------
# 3. CHECK DATA
# ---------------------------
print("\n📊 Category Distribution:\n")
print(df['Category'].value_counts())

# ---------------------------
# 4. FIX CLASS IMBALANCE via oversampling minority classes
# ---------------------------
max_size = df['Category'].value_counts().max()
MIN_SAMPLES = 80  # oversample anything below this

balanced_dfs = []
for category, group in df.groupby('Category'):
    if len(group) < MIN_SAMPLES:
        group = resample(group, replace=True, n_samples=MIN_SAMPLES, random_state=42)
    balanced_dfs.append(group)

df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
print(f"\n✅ Balanced dataset size: {len(df_balanced)} rows")
print(df_balanced['Category'].value_counts())

# ---------------------------
# 5. FEATURES & LABELS
# ---------------------------
X = df_balanced['cleaned']
y = df_balanced['Category']

# ---------------------------
# 6. TF-IDF
# ---------------------------
tfidf = TfidfVectorizer(
    max_features=8000,       # more features = better discrimination
    stop_words='english',
    ngram_range=(1, 3),      # trigrams catch role-specific phrases
    sublinear_tf=True,       # log normalization reduces impact of frequent terms
    min_df=2,                # ignore terms that appear in < 2 docs
)

X_vec = tfidf.fit_transform(X)

# ---------------------------
# 7. TRAIN TEST SPLIT (stratified to keep class ratio)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42,
    stratify=y              # ensures each split has same class distribution
)

# ---------------------------
# 8. MODEL
# ---------------------------
model = LinearSVC(
    class_weight='balanced',  # penalizes errors on minority classes more
    max_iter=2000,
    C=1.0
)
model.fit(X_train, y_train)

# ---------------------------
# 9. EVALUATION
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Highlight weak classes
report = classification_report(y_test, y_pred, output_dict=True)
print("\n⚠️  Weak Classes (F1 < 0.60):")
for label, metrics in report.items():
    if isinstance(metrics, dict) and metrics.get('f1-score', 1) < 0.60:
        print(f"   {label}: F1={metrics['f1-score']:.2f}, Support={int(metrics['support'])}")

# ---------------------------
# 10. SAVE
# ---------------------------
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
print("\n✅ Model trained and saved successfully!")
