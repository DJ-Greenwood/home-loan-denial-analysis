from data_processing import clean_data, process_data

from exploratory_analysis import exploratory_data_analysis as detailed_eda
import pandas as pd

def main():

    # Step 1: Clean and process the data
    print("Cleaning and processing data...")
    raw_data_path = "../data/raw/loan_data_train.csv"
    df = pd.read_csv(raw_data_path)
    clean_df = clean_data(df)
    processed_df = process_data(clean_df)
    processed_df.to_csv("../data/processed/cleaned_loan_data.csv", index=False)
    print("Data cleaned and processed. Saved to data/processed/cleaned_loan_data.csv")

    # Step 2: Perform exploratory data analysis
    print("Performing exploratory data analysis...")
    detailed_eda(processed_df)
    print("Exploratory data analysis completed.")

    # Step 3: Perform detailed exploratory data analysis
    print("Performing detailed exploratory data analysis...")
    detailed_eda(processed_df)
    print("Detailed exploratory data analysis completed.")

if __name__ == "__main__":
    main()
