import pandas as pd
import os


def save_result(path, result):

    file_exists = os.path.isfile(path)

    df = pd.DataFrame([result])

    df.to_csv(path, mode="a", header=not file_exists, index=False)
