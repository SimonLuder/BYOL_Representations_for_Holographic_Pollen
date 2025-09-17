import os
import pandas as pd
from pollen_datasets.poleno import DataSetup


def setup_data():

    setup = DataSetup()

    # Download raw data if not already present
    if not os.path.exists("./data/raw/poleno/"):
        os.makedirs("./data/raw/poleno/")
        setup.download_tables_from_db(
            db_path="Z:/marvel/marvel-fhnw/data/Poleno/poleno_marvel_old.db",
            csv_dir="./data/raw/poleno/")

    # Preprocess raw data and save to processed folder
    if not os.path.exists("./data/processed/poleno/"):
        os.makedirs("./data/processed/poleno/")
        df = pd.read_csv("./data/raw/poleno/computed_data_full.csv")
        df = setup.preprocess(df)
        df.to_csv("./data/processed/poleno/computed_data_full.csv", index=False)


if __name__ == "__main__":
    setup_data()