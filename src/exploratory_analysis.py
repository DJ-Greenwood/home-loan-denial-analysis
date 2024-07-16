import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(df):
    # Implement EDA steps here
    sns.pairplot(df)
    plt.show()

if __name__ == "__main__":
    data_path = "data/processed/home_loan_data_processed.csv"
    df = pd.read_csv(data_path)
    exploratory_data_analysis(df)
