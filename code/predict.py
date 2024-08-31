import pickle
import pandas as pd
import json
import fire
from typing import Union
import numpy as np

import os
import sys
from pathlib import Path

filepath = Path(__file__).resolve().parent
esm_code = filepath.parent.joinpath("esm_embeddings/ml")
sys.path.append(str(esm_code))

from dataloader_v1 import process_dataset

def dataset2embeddings(input_csv: Union[str, os.PathLike], seq_col: str, batch_size: int) -> np.ndarray:
    df = pd.read_csv(input_csv)
    seqs = df[seq_col].values
    tmp_df = pd.DataFrame(None)
    tmp_df["sequence"] = seqs
    tmp_df["mean_pH"] = 0
    embeddings, _ = process_dataset(tmp_df, batch_size=batch_size)
    return embeddings

def predict(
    input_csv: Union[str, os.PathLike],
    id_col: str,
    seq_col: str,
    model_fname: Union[str, os.PathLike],
    output_csv: Union[str, os.PathLike],
    batch_size: int = 64
) -> None:

    # load model
    with open(model_fname, "rb") as fin:
        model = pickle.load(fin)

    # Read input CSV
    df = pd.read_csv(input_csv)

    if model.__class__.__name__ == 'kmers':
        embeddings = df[seq_col].values
    else:
        embeddings = dataset2embeddings(input_csv, seq_col, batch_size)

    # make predictions
    predictions = model.predict(embeddings)

    # Create output dataframe
    output_df = pd.DataFrame({
        id_col: df[id_col],
        'Sequence': df[seq_col],
        'Predicted_pH': predictions.flatten()
    })

    # Save predictions
    output_df.to_csv(output_csv, index=False)

    return

if __name__ == "__main__":
    fire.Fire(predict)
