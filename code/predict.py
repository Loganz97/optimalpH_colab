import pickle
import pandas as pd
import json
import argparse
from typing import Union
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import os
import sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from datetime import datetime

filepath = Path(__file__).resolve().parent
esm_code = filepath.parent.joinpath("esm_embeddings/ml")
sys.path.append(str(esm_code))

try:
    from dataloader_v1 import process_dataset
except ImportError as e:
    print(f"Error importing process_dataset: {str(e)}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def get_model(model_name):
    if os.path.exists(model_name):
        return model_name
    else:
        try:
            model_id = "Loganz97/optimalpH"
            local_path = hf_hub_download(repo_id=model_id, filename=f"weights/{model_name}")
            return local_path
        except Exception as e:
            print(f"Error downloading model from Hugging Face: {str(e)}")
            print("Please ensure you have provided a valid local file path or a correct model name.")
            sys.exit(1)

def predict(
    input_csv: Union[str, os.PathLike],
    model: Union[str, os.PathLike],
    output_csv: Union[str, os.PathLike],
    id_col: str = 'ID',
    seq_col: str = 'Sequence'
) -> None:

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
        model_path = get_model(model)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            with open(model_path, "rb") as fin:
                model = pickle.load(fin)

        df = pd.read_csv(input_csv)

        if id_col not in df.columns or seq_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{id_col}' and '{seq_col}' columns.")

        embeddings = process_dataset(df[[seq_col]])
        predictions = model.predict(embeddings)

        output_df = pd.DataFrame({
            id_col: df[id_col],
            'Sequence': df[seq_col],
            'Predicted_pH': predictions.flatten()
        })

        output_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict optimal pH for protein sequences.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--model", required=True, 
                        help="Model to use for prediction. Options:\n"
                             "- model_xgboost: XGBoost model (most accurate, MAE ~0.6)\n"
                             "- model_knn: K-Nearest Neighbors model (MAE ~0.7)\n"
                             "- model_kmers: K-mers based model (MAE ~0.9)\n"
                             "You can also provide a path to a local model file.")
    parser.add_argument("--output_csv", help="Path to output CSV file")
    parser.add_argument("--id_col", default="ID", help="Name of the ID column (default: ID)")
    parser.add_argument("--seq_col", default="Sequence", help="Name of the sequence column (default: Sequence)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Generate default output filename if not provided
    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"optimalpH_predictions_{timestamp}.csv"
    
    predict(
        input_csv=args.input_csv,
        model=args.model,
        output_csv=args.output_csv,
        id_col=args.id_col,
        seq_col=args.seq_col
    )
