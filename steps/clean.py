import pandas as pd
import yaml


class Cleaner:

    def __init__(self):
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)

        self.target_column = config["data"]["target_column"]

    def clean_data(self, df: pd.DataFrame):

        # Drop missing values
        df = df.dropna().reset_index(drop=True)

        # Convert target column to numeric
        if self.target_column in df.columns:
            df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce")

        df = df.dropna(subset=[self.target_column])

        return df


if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    cleaner = Cleaner()

    clean_train = cleaner.clean_data(train)
    clean_test = cleaner.clean_data(test)

    clean_train.to_csv("data/clean_train.csv", index=False)
    clean_test.to_csv("data/clean_test.csv", index=False)

    print("Data cleaning completed successfully")
