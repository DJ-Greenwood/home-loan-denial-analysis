import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(df):
    df = df.drop("Loan_ID", axis=1)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
   
    data_path = "../data/processed/cleaned_loan_data.csv"
    df = pd.read_csv(data_path)
    train_model(df)
