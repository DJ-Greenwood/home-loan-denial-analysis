import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Convert categorical variables to numeric codes
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Convert 'Dependents' to numeric, handle '3+' case
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    return df


from sklearn.preprocessing import StandardScaler

def process_data(df):
    # Feature Engineering: Create Debt-to-Income Ratio
    df['Debt_to_Income_Ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + df['CoapplicantIncome'] + 1)
    
    # Scaling numerical features
    scaler = StandardScaler()
    df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Debt_to_Income_Ratio']] = scaler.fit_transform(
        df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Debt_to_Income_Ratio']])
    
    return df


if __name__ == "__main__":
    raw_data_path = "data/raw/loan_data_train.csv"
    processed_data_path = "data/processed/loan_data_processed_test.csv"

    df = load_data(raw_data_path)
    df_clean = clean_data(df)
    df_processed = process_data(df_clean)
    df_processed.to_csv(processed_data_path, index=False)
