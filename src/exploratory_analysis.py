import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def exploratory_data_analysis(df):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Histograms and Density Plots for numeric features
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Debt_to_Income_Ratio']
    df[num_features].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Histograms of Numeric Features', size=20)
    plt.show()

    df[num_features].plot(kind='density', subplots=True, layout=(3, 2), sharex=False, figsize=(15, 10))
    plt.suptitle('Density Plots of Numeric Features', size=20)
    plt.show()

    # Box Plots for numeric features
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df[num_features])
    plt.title('Box Plots of Numeric Features', size=20)
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap', size=20)
    plt.show()

    # Count Plots for categorical features
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(cat_features, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=df, x=feature)
        plt.title(f'Count Plot of {feature}', size=15)
    plt.tight_layout()
    plt.show()

    # Pairplot for relationships between features
    sns.pairplot(df[num_features + ['Loan_Status']], hue='Loan_Status')
    plt.show()

if __name__ == "__main__":
    file_path = "data/processed/loan_data_processed_test.csv"
    df = pd.read_csv(file_path)
    
    exploratory_data_analysis(df)
