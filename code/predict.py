import pickle
import pandas as pd
import json
import fire
from typing import Union
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import os
import sys
from pathlib import Path
import torch

filepath = Path(__file__).resolve().parent
esm_code = filepath.parent.joinpath("esm_embeddings/ml")
sys.path.append(str(esm_code))

try:
    from dataloader_v1 import process_dataset
except ImportError as e:
    print(f"Error importing process_dataset: {str(e)}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def predict(
    input_csv: Union[str, os.PathLike],
    id_col: str,
    seq_col: str,
    model_fname: Union[str, os.PathLike],
    output_csv: Union[str, os.PathLike]
) -> None:

    try:
        # Force CUDA device if available
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
        # load model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            with open(model_fname, "rb") as fin:
                model = pickle.load(fin)

        # Read input CSV
        df = pd.read_csv(input_csv)

        # Process dataset
        embeddings, _ = process_dataset(df[[seq_col]])

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

        print(f"Predictions saved to {output_csv}")

    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(predict)
