# ===============================
# 🧪 1. Load & Basic Exploration
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub # download dataset

# Download latest version
path = kagglehub.dataset_download("radheshyamkollipara/bank-customer-churn")

print("Path to dataset files:", path)

# download into pandas -> Convert raw files into tables that the program can use.
import os # See what files are in the folder path.

print(os.listdir(path)) #['Customer-Churn-Records.csv']
df = pd.read_csv(f"{path}/Customer-Churn-Records.csv")  # download into  DataFrame
print(df.head())   # View the first 5 rows of data -> () = default is 5 ; if want to view 10 (10)
print(df.info())   # See data types Structure and missing

pd.set_option('display.max_columns', None) #Show all columns no...
print(df.describe()) #View basic statistics for all numeric columns.



# ======================================================
# 🧠 EDA
# 🎯 Find out which column "affect" staying/exited
# ======================================================

# 📊 2. Categorical Analysis
# ===============================

# The column that is “text” No number,mean,max,min
categorical_cols = ['Geography', 'Gender', 'Card Type']

for col in categorical_cols:
    print(f"\n distribution of {col} Sort by Exited")
    print(df.groupby([col, 'Exited']).size())


# 📈 3. Numeric Analysis
# ===============================

# Let's look at the mean, max, and min values of the important columns to see how they affect usability.
# Create a list of columns; want to view.
cols = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Exited', 'Satisfaction Score', 'Card Type','Complain','Point Earned'
]

# For numeric columns: Show min, max, mean separated by Exited.
numeric_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score','Complain', 'Point Earned'
]

print("📊 สถิติของคอลัมน์ตัวเลข (numeric)")
print(df.groupby('Exited')[numeric_cols].agg(['min','max','mean']))


# 🔥 4. Data Visualization
# ===============================

# Graph the data to make it easier to decide on column selection.
os.makedirs("charts", exist_ok=True)

# 🟡 STEP 1 : All columns
all_features = [
    'Gender', 'Geography', 'Card Type',     # categorical
    'Age', 'Balance', 'CreditScore',        # numeric
    'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Complain', 'Satisfaction Score', 'Point Earned'
]

# 📊 STEP 2 : Visualization
for col in all_features:
    plt.figure(figsize=(8,5))
    if df[col].dtype == 'object' or df[col].nunique() <= 10:
        # It is group information. → countplot
        sns.countplot(data=df, x=col, hue='Exited')
        plt.title(f'{col} vs Exited (count)')
    else:
        # is a continuous number. → histplot
        sns.histplot(data=df, x=col, hue='Exited', kde=True, multiple='stack')
        plt.title(f'Distribution of {col} by Exited')
    plt.tight_layout()
    plt.savefig(f'charts/{col.replace(" ","_").lower()}_viz.png', dpi=300)
    plt.close()

print("✅ Step 2 done: Auto charts created for all features")


# 📈 STEP 3 : Exit Rate (Look at the % of people exited.)
# → For all columns with number unique not more than 10
# Because some columns have a wide range of values, using Exit Rate would create a graph that’s hard to read.
cat_like = [c for c in all_features if df[c].nunique() <= 10]
for col in cat_like:
    rate = (pd.crosstab(df[col], df['Exited'], normalize='index') * 100).rename(columns={0:'Stay_%',1:'Exit_%'})
    ax = rate.sort_values('Exit_%')['Exit_%'].plot(kind='barh', figsize=(8,5))
    ax.set_xlabel('Exit rate (%)')
    ax.set_title(f'Exit rate by {col}')
    plt.tight_layout()
    plt.savefig(f'charts/exit_rate_{col.replace(" ","_").lower()}.png', dpi=300)
    plt.close()

print("✅ Step 3 done: Exit-rate charts created")
print("📸 All charts saved in 'charts' folder.")


# ======================================================
# 🧠 Data Preparation
# 🎯 Verify that the dataset is ready for preprocessing.
# ======================================================

print("\n[Prep] Missing values per column:")
print(df.isna().sum())

print("\n[Prep] Duplicate rows:", df.duplicated().sum())

print("\n[Prep] Dtypes:")
print(df.dtypes)

# NO Missing values, Duplicate rows

# 👉 Select the columns to use for modeling (11 features)
TARGET = 'Exited'
FEATURES = [
    'Gender', 'Geography', 'Age', 'Balance', 'CreditScore',
    'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Point Earned'
]

# Confirm that the columns are complete.
# To check if there are any missing/wrongly named columns.
missing_cols = [c for c in FEATURES+[TARGET] if c not in df.columns]
assert len(missing_cols) == 0, f"Columns not found: {missing_cols}"
# The results show nothing.Passed.✅



# ======================================================
# 🧠 Data Preprocessing
# 🎯 Encode (categorical), Scale (numeric), Train/Test split
# ======================================================
from sklearn.model_selection import train_test_split   # 👉 for dividing data train / test set.
from sklearn.preprocessing import OneHotEncoder, StandardScaler   # 👉 OneHot: Convert category data to numbers, StandardScaler: Scale numbers
from sklearn.compose import ColumnTransformer         # 👉 Manage multiple columns simultaneously
from sklearn.pipeline import Pipeline                 # 👉 Make the “conveyor belt” of data and model transformation work in one step.

# 4.1 Separate X, y  for teach model
# X = input data (features) that the model will use to learn.
# y = output data (targets) that the model will try to predict.
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# 4.2 Specify the column type
categorical_cols = ['Gender', 'Geography']     # as text
numeric_cols = [c for c in FEATURES if c not in categorical_cols]  # The rest are numbers.



# ============================================
# ✨ HIGHLIGHT ; Add engineered features (MY Dimension 1 & 2 & 3 )
# ============================================

# ---- (Dimension 1 & 2  ⭐) Features by meaning & Relationships between features. ----
X['Engagement_Score'] = (
    X['NumOfProducts'] + X['HasCrCard'] + X['IsActiveMember']
    + (X['CreditScore'] / 1000.0) + (X['Point Earned'] / 1000.0)
)
X['Loyalty_Score'] = X['Tenure'] + 2 * X['IsActiveMember']
X['Financial_Score'] = (
    (X['Balance'] / 100000.0) + (X['EstimatedSalary'] / 100000.0)
    + (X['Point Earned'] / 1000.0)
)
extra_engineered = ['Engagement_Score','Loyalty_Score','Financial_Score']
numeric_cols = numeric_cols + extra_engineered

# ---- (Dimension 3 ⭐) Binning for widely distributed numeric. ----
ENABLE_BINNING = True
if ENABLE_BINNING:
    # Arrange the ranges to make it easier to see the pattern (and plan to use it in the model).
    X['Age_bin'] = pd.cut(
        X['Age'], bins=[0,25,35,45,55,65,200],
        labels=['<25','25-35','35-45','45-55','55-65','65+']
    )
    X['Balance_bin'] = pd.cut(
        X['Balance'], bins=[-1, 50_000, 150_000, 1e12],
        labels=['Low','Mid','High']
    )
    X['Salary_bin'] = pd.cut(
        X['EstimatedSalary'], bins=[-1, 60_000, 120_000, 1e12],
        labels=['Low','Mid','High']
    )
    X['Point_bin'] = pd.cut(
        X['Point Earned'], bins=[-1, 400, 800, 10_000],
        labels=['Low','Mid','High']
    )
    X['Tenure_bin'] = pd.cut(
        X['Tenure'], bins=[-1, 2, 6, 10],
        labels=['Short (0-2)', 'Medium (3-6)', 'Long (7-10)']
    )


    # Add bin as an additional category.
    categorical_cols += ['Age_bin','Balance_bin','Salary_bin','Point_bin','Tenure_bin']

# 4.3 Converter: one-hot for categories(Change the text to the number 0/1), standardize for numbers(Scale the numbers to the same scale.)
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

scaler = StandardScaler()

preprocessor = ColumnTransformer(    # Combine both models in one step before submitting to the model.
    transformers=[
        ('cat', ohe, categorical_cols),
        ('num', scaler, numeric_cols)
    ],
    remainder='drop'
)

# 4.4 separate train/test (Maintain proportions churn by stratify.)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[Split] X_train: {X_train.shape}, X_test: {X_test.shape}") # Data size
print("[Split] y_train ratio:\n", y_train.value_counts(normalize=True).round(3)) # proportion 0/1 in train
print("[Split] y_test  ratio:\n", y_test.value_counts(normalize=True).round(3))  # proportion 0/1 in test

# 4.5 create pipeline; Prepare data
prep_pipeline = Pipeline(steps=[('prep', preprocessor)])

# fit only train and transform both train/test
X_train_ready = prep_pipeline.fit_transform(X_train)
X_test_ready  = prep_pipeline.transform(X_test)

print(f"\n[Ready] X_train_ready shape: {X_train_ready.shape}")
print(f"[Ready] X_test_ready  shape: {X_test_ready.shape}")

# 4.6 See the feature name after conversion.
try:
    cat_names = prep_pipeline.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(cat_names) + numeric_cols
    print("\n[Ready] Example feature names (first 25):")
    print(feature_names[:25])
except Exception as e:
    print("\n[Ready] Could not fetch feature names:", e)

print("\n✅ Data Preparation checked.")
print("✅ Data Preprocessing completed. Ready for modeling!")


# ======================================================
# 🧠 Model Building & Evaluation
# 🎯 Train and evaluate models → to predict the risk of customer churn
# ======================================================

# ===============================
# 1) Train two models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Enter the prepared data into these two models.
logit = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logit.fit(X_train_ready, y_train)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=4,
    class_weight='balanced', random_state=42
)
rf.fit(X_train_ready, y_train)

# ===============================
# 2) Evaluate (Accuracy / Precision / Recall / F1)

def eval_model(name, model):
    y_pred = model.predict(X_test_ready)
    print(f"\n===== {name} =====")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall   :", round(recall_score(y_test, y_pred), 3))
    print("F1-score :", round(f1_score(y_test, y_pred), 3))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

eval_model("LogisticRegression", logit)
eval_model("RandomForest", rf)


# ===============================
# 3) Explain per-customer (Logistic Regression)
# It can tell: Which features have the greatest influence on the model's decisions.
# If this customer is predicted to “exited”, it can say “why”.

# Prepare feature names after preprocessing (if feature_names doesn't exist, create a simple one)
if 'feature_names' not in globals() or feature_names is None:
    feature_names = [f"f{i}" for i in range(X_train_ready.shape[1])]

coef = logit.coef_.ravel()  # The weight of each feature (positive values push it "out", negative values push it "stay").

def explain_one(sample_idx=0, top_k=3):
    """
    Describe each customer individually X_test_ready[row]
    """
    x = X_test_ready[sample_idx]
    contrib = x * coef  # feature contribution
    order = np.argsort(contrib)[::-1]
    top_idx = [i for i in order if contrib[i] > 0][:top_k]
    reasons = [(feature_names[i], float(contrib[i])) for i in top_idx]
    proba = float(logit.predict_proba(X_test_ready[[sample_idx]])[0, 1])
    return proba, reasons

# Try explaining to the first customer in test
p, reasons = explain_one(sample_idx=0, top_k=2)
print("\n[Explain] P(exit) for sample #0:", round(p, 3))
print("[Explain] Top reasons (feature, contribution):", reasons)

# ===============================
# ✨ HIGHLIGHT Targeted Action Playbook (for LogisticRegression model)
#    - map Feature Name → Personalized Proactive Offers

action_map = {
    'Geography_Germany': "เสนอบริการเฉพาะประเทศ (ภาษาท้องถิ่น/ช่องทางสาขา)",
    'Age_bin_25-35': "โปรโมชันดิจิทัล/โมบายแบงก์กิ้งสำหรับวัยทำงานต้น",
    'Age_bin_45-55': "แผนที่ปรึกษาการเงิน/สินเชื่อบ้าน รีไฟแนนซ์",
    'Balance_bin_Low': "โปรโมชันค่าโอน/ค่าธรรมเนียมต่ำ เพื่อกระตุ้นการใช้งาน",
    'Balance_bin_High': "โปรแกรมความภักดีระดับพรีเมียม / ผู้จัดการความสัมพันธ์ (RM)",
    'Salary_bin_Low': "สินเชื่อดอกเบี้ยต่ำ/ผ่อนชำระยืดหยุ่น",
    'Salary_bin_High': "การลงทุน/กองทุน/บัตรพรีเมียม",
    'Point_bin_Low': "แคมเปญสะสมแต้ม x2 สำหรับหมวดที่ลูกค้าสนใจ",
    'Point_bin_High': "ของรางวัลระดับสูง/อัปเกรดสถานะสมาชิก",
    'Tenure_bin_Short (0-2)': "อบรม onboarding + ข้อเสนอ welcome เฉพาะบุคคล",
    'Tenure_bin_Medium (3-6)': "รีวิวผลิตภัณฑ์ที่ยังไม่ได้ใช้ + cross-sell ที่เกี่ยวข้อง",
    'Tenure_bin_Long (7-10)': "โปรแกรมขอบคุณสมาชิกเก่า/ค่าธรรมเนียมพิเศษ",
    'NumOfProducts': "เสนอผลิตภัณฑ์เสริมที่สอดคล้องพฤติกรรม",
    'HasCrCard': "แคมเปญใช้จ่ายรายหมวดเพื่อเพิ่มประสบการณ์บัตร",
    'IsActiveMember': "แผน active re-engagement (แจ้งเตือนอัจฉริยะ/ภารกิจรายสัปดาห์)",
    'CreditScore': "ให้คำปรึกษาปรับปรุงเครดิต + ทางเลือกสินเชื่อเหมาะสม",
}

def suggest_actions(reasons, top_m=2):
    picked = []
    for fname, _ in reasons:
        #The one-hot name will be in the format prefix_xxx → find the matching key in startswith.
        key = None
        for k in action_map.keys():
            if fname.startswith(k):
                key = k; break
        if key and action_map[key] not in picked:
            picked.append(action_map[key])
        if len(picked) >= top_m:
            break
    # fallback If there is no plan for handling that specific feature → the system will offer a “Basic Plan” instead.
    if not picked:
        picked = ["โทรเช็กความพึงพอใจและปัญหาเร่งด่วน + เสนอโปรเฉพาะพฤติกรรมล่าสุด"]
    return picked

print("\n[Action] Suggested actions for sample #0:")
print(suggest_actions(reasons, top_m=2))



