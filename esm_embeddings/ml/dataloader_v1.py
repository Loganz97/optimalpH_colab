import pandas as pd
import numpy as np

import torch
import esm
import fire

from scipy.special import softmax
from scipy.stats import entropy
# from sklearn.decomposition import PCA
# 
import gc

from tqdm import tqdm
from time import time

import argparse
import os

def load_esm_model():
    token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10,
             'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}

    t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    return t_model, batch_converter

def prepare_batches(sequences, max_tokens_per_batch):
    sizes = [(len(s), i) for i, s in enumerate(sequences)]
    sizes.sort()
    batches = []
    buf = []
    max_len = 0

    def _flush_current_buf():
        nonlocal max_len, buf
        if len(buf) == 0:
            return
        batches.append(buf)
        buf = []
        max_len = 0

    for sz, i in sizes:
        if max(sz, max_len) * (len(buf) + 1) > max_tokens_per_batch:
            _flush_current_buf()
        max_len = max(max_len, sz)
        buf.append(i)
    _flush_current_buf()

    return batches

def get_esm_embeddings_batched(sequences, model, batch_converter, max_tokens_per_batch=10000):
    model.eval()
    batches = prepare_batches(sequences, max_tokens_per_batch)
    data = [(i, s) for i, s in enumerate(sequences)]

    data_loader = torch.utils.data.DataLoader(
        data, collate_fn=batch_converter, batch_sampler=batches
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Embeddings will be calculated with device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    embeddings = np.zeros(shape=(len(data), 1280))
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(batches)):
            toks = toks.to(device)
            results = model(toks, repr_layers=[33])
            results = results["representations"][33].detach().cpu().numpy()

            for r, s, l in zip(results, strs, labels):
                embeddings[l] = r[1:len(s)+1].mean(0)

            if (batch_idx % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    return embeddings

# Remove PCA reduction for now but might be interesting to use for a clustering scheme to visualize model labeled embeddings
#def PCA_reduction(embeddings_train, embeddings_test, variance_threshold=0.99):
#    pca = PCA()
#    pca.fit(embeddings_train)
#    
#    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
#    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
#    
#    print(f"Number of PCA components for {variance_threshold*100}% variance: {n_components}")
#    
#    pca_final = PCA(n_components=n_components)
#    pca_final.fit(embeddings_train)
#    
#    embeddings_train_reduced = pca_final.transform(embeddings_train)
#    embeddings_test_reduced = pca_final.transform(embeddings_test)
#    
#    return embeddings_train_reduced, embeddings_test_reduced, pca_final

def process_dataset(df):
    model, batch_converter = load_esm_model()
    sequences = df['Sequence'].values
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    
    # Perform PCA reduction (might be useful for clustering)
    #embeddings_reduced, _, pca_model = PCA_reduction(embeddings, embeddings)
    
    #return embeddings_reduced, pca_model

def main(input_csv, seq_col, output_emb):
    df = pd.read_csv(input_csv)
    sequences = df[seq_col].values.tolist()

    model, batch_converter = load_esm_model()

    start = time()
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    finish = time()

    print(f"Time elapsed: {finish - start}")

    # Perform PCA reduction (might be useful for clustering)
    # embeddings_reduced, _, pca_model = PCA_reduction(embeddings, embeddings)

    X = np.zeros(len(embeddings_reduced),
                 dtype=[('X', 'f4', (embeddings_reduced.shape[1],))])
    X['X'] = embeddings_reduced
    np.save(output_emb, X)

    print(f"Reduced embeddings saved to {output_emb}")

if __name__ == "__main__":
    fire.Fire(main)
