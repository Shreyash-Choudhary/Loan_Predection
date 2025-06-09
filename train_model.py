import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from explainerdashboard import ClassifierExplainer
import os

print("Current working directory:", os.getcwd())

print("Loading dataset...")
try:
    df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)

print("Preprocessing data...")
df.dropna(inplace=True)
if 'Loan_ID' in df.columns:
    df.drop(['Loan_ID'], axis=1, inplace=True)

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Saving model and explainer...")
joblib.dump(model, "model.pkl")

explainer = ClassifierExplainer(model, X_test, y_test, labels=["Not Approved", "Approved"])
explainer.dump("explainer.pkl")

print("Training complete! Model and explainer saved.")
