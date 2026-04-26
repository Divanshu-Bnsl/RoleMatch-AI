import pandas as pd
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# ---- same clean_text ----
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---- load and prepare data ----
df = pd.read_csv("data/resume.csv")
df = df[['Resume_str', 'Category']]
df.rename(columns={'Resume_str': 'Resume'}, inplace=True)
df['cleaned'] = df['Resume'].apply(clean_text)

# ---- same balancing as train.py ----
MIN_SAMPLES = 80
balanced_dfs = []
for category, group in df.groupby('Category'):
    if len(group) < MIN_SAMPLES:
        group = resample(group, replace=True, n_samples=MIN_SAMPLES, random_state=42)
    balanced_dfs.append(group)
df = pd.concat(balanced_dfs).reset_index(drop=True)

# ---- load tfidf ----
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

X = tfidf.transform(df['cleaned'])
y = df['Category']
classes = sorted(y.unique())

# ---- binarize labels ----
y_bin = label_binarize(y, classes=classes)

# ---- train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42
)

# ---- train calibrated model (needed for predict_proba) ----
base = LinearSVC(class_weight='balanced', max_iter=2000, C=1.0)
model = CalibratedClassifierCV(base, cv=3)
ovr   = OneVsRestClassifier(model)
ovr.fit(X_train, y_train)

y_score = ovr.predict_proba(X_test)

# ---- compute ROC per class ----
fpr, tpr, roc_auc = {}, {}, {}
for i, cls in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ---- macro average ----
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(classes)
macro_auc = auc(all_fpr, mean_tpr)

# ---- plot ----
fig, axes = plt.subplots(5, 5, figsize=(22, 20))
axes = axes.flatten()

for i, cls in enumerate(classes):
    ax = axes[i]
    ax.plot(fpr[i], tpr[i], color='darkorange', lw=2,
            label=f'AUC = {roc_auc[i]:.2f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(cls, fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('FPR', fontsize=7)
    ax.set_ylabel('TPR', fontsize=7)
    ax.tick_params(labelsize=7)

# ---- macro average plot in last cell ----
ax = axes[len(classes)]
ax.plot(all_fpr, mean_tpr, color='navy', lw=2,
        label=f'Macro AUC = {macro_auc:.2f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_title('MACRO AVERAGE', fontsize=9, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlabel('FPR', fontsize=7)
ax.set_ylabel('TPR', fontsize=7)
ax.tick_params(labelsize=7)

# hide unused subplots
for j in range(len(classes) + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('ROC Curves — Resume Skill Gap Analyzer (One vs Rest)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ ROC curve saved as roc_curve.png")
print(f"\n📊 AUC Scores:")
for i, cls in enumerate(classes):
    print(f"   {cls:<25} AUC = {roc_auc[i]:.3f}")
print(f"\n   {'MACRO AVERAGE':<25} AUC = {macro_auc:.3f}")