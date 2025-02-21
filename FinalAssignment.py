# =======================================
# 1. IMPORT LIBRARIES
# =======================================
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm  # Progress bar

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from transformers import AutoTokenizer, AutoModel
import shap
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing dataset

# =======================================
# 2. LOAD AND SAMPLE THE DATA
# =======================================
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Reduce data to first 4000 rows for faster execution
df = df.head(4000)

print("Data sample:\n", df.head(), "\n")

# =======================================
# 3. DEFINE THE TARGET
# =======================================
df["is_stolen"] = (df["Crm Cd Desc"] == "VEHICLE - STOLEN").astype(int)

# =======================================
# 4. BUILD A TEXT FEATURE
# =======================================
df["text_feature"] = (
    df["AREA NAME"].fillna("")
    + " "
    + df["Premis Desc"].fillna("")
    + " "
    + df["Status"].fillna("")
    + " "
    + df["LOCATION"].fillna("")
)

# =======================================
# 5. SPLIT TRAIN/TEST
# =======================================
STUDENT_ID = 296726  # Random state for reproducibility

X = df["text_feature"]
y = df["is_stolen"]

# Ensure balanced labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=STUDENT_ID, stratify=y if len(y.unique()) > 1 else None
)

# =======================================
# 6. LOAD TRANSFORMER MODEL
# =======================================
MODEL_NAME = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# =======================================
# 7. FUNCTION TO GENERATE EMBEDDINGS (BATCHED)
# =======================================
def get_embeddings(texts, batch_size=32):
    embeddings = []
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)

    print("Embedding generation complete!")
    return np.array(embeddings)

# =======================================
# 8. GENERATE EMBEDDINGS
# =======================================
X_train_emb = get_embeddings(X_train.tolist())
X_test_emb = get_embeddings(X_test.tolist())

# =======================================
# 9. APPLY SMOTE TO BALANCE CLASSES
# =======================================
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=STUDENT_ID)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_emb, y_train)

print(f"Balanced dataset: {np.bincount(y_train_balanced)}")  # Check class distribution

# =======================================
# 10. TRAIN LOGISTIC REGRESSION MODEL ON BALANCED DATA
# =======================================
lr_model = LogisticRegression(random_state=STUDENT_ID)
lr_model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred = lr_model.predict(X_test_emb)

# Evaluate
f1 = f1_score(y_test, y_pred, pos_label=1)
print("F1-score (is_stolen=1) after SMOTE:", f1)

# Full classification report
print("\nClassification Report After SMOTE:")
print(classification_report(y_test, y_pred))

# =======================================
# 11. EXPLAIN MODEL WITH SHAP
# =======================================
explainer = shap.LinearExplainer(lr_model, X_train_balanced)
shap_values = explainer(X_test_emb)

# Global feature importance
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test_emb)

# Local explanation for first test sample
if X_test_emb.shape[0] > 0:
    print("Generating SHAP waterfall plot for first test sample...")
    shap.plots.waterfall(shap_values[0], max_display=10)
