import pandas as pd
import numpy as np

import torch
try:
    import esm
except ImportError:
    print("Error: ESM module not found. Please install it using 'pip install fair-esm' or 'pip install git+https://github.com/facebookresearch/esm.git'")
    sys.exit(1)

import fire

from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA

import gc

from tqdm import tqdm
from time import time

import argparse
import os

try:
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.nn.wrap import enable_wrap, wrap
except ImportError:
    print("Warning: fairscale not found. FSDP functionality will not be available.")

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap



def load_esm_model_v2():
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    # download model data from the hub
    model_name = "esm2_t33_650M_UR50D"
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

    # token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10,
            #  'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}

    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=True,  # enable cpu offloading
    )
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, vocab = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )
        batch_converter = vocab.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)

    return model, batch_converter

def load_esm_model():
    try:
        token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10,
                     'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}

        t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()

        return t_model, batch_converter
    except Exception as e:
        print(f"Error loading ESM model: {str(e)}")
        raise

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


def load_dataset(fname):


    print('START DATALOADER')

    # STAY WITH SINGLE DATAFRAME
    # MAKE QUALITY CHECK

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', '-i', type=str, help='input .csv file with the required_columns', default='./data/file1.csv')
    # args = parser.parse_args()

    # REQUIRED COLUMNS IN THE DATAFRAME
    required_columns = ['sequence', 'mean_pH']

    # if os.path.exists(args.input):
        # if args.input.endswith('.csv'):
            # input_dir, input_fn = os.path.split(args.input)
        # else:
            # raise ValueError(f'The input file should be .csv : {args.input}')
    # else:
        # raise ValueError(f'No such file: {args.input}')


    df = pd.read_csv(fname)
    if not set(required_columns).issubset(df.columns.to_list()):
        print('required columns: ', required_columns)
        print(df.columns)
        raise ValueError('the input dataframe does not contain required columns')

    print('END DATALOADER')

    return df


def get_esm_embeddings_batched(sequences, model, batch_converter, max_tokens_per_batch=10000, use_cuda=True):
    model.eval()
    batches = prepare_batches(sequences, max_tokens_per_batch) #group sequences of the comparable length into batches
    data = [(i, s) for i, s in enumerate(sequences)] #assing index to each sequence

    data_loader = torch.utils.data.DataLoader(
        data, collate_fn=batch_converter, batch_sampler=batches
    )

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    print(f"Embeddings will be calculated with device: {device}")

    embeddings = np.zeros(shape=(len(data), 1280)) #1280 is embedding dimension
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(batches)):

            toks = toks.to(device)
            results = model(toks, repr_layers=[33])
            logits = (results['logits'].detach().cpu().numpy()[0,].T)[4:24,1:-1]
            probability = softmax(logits,axis=0)
            results = results["representations"][33].detach().cpu().numpy()

            for r, s, l in zip(results, strs, labels):
                embeddings[l] = r[1:len(s)+1].mean(0)

            if (batch_idx % 10) == 0:

                del toks
                del results
                del probability
                del logits
                gc.collect(); torch.cuda.empty_cache()

    # del toks
    # del results
    # del probability
    # del logits
    gc.collect(); torch.cuda.empty_cache()

    return embeddings

def PCA_reduction(embeddings_train, embeddings_test, variance_threshold=0.99):
    # Initialize PCA without specifying the number of components
    pca = PCA()
    
    # Fit PCA on the training data
    pca.fit(embeddings_train)
    
    # Calculate the cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components that explain at least 99% of the variance
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    
    print(f"Number of PCA components for {variance_threshold*100}% variance: {n_components}")
    
    # Create a new PCA object with the determined number of components
    pca_final = PCA(n_components=n_components)
    pca_final.fit(embeddings_train)
    
    # Transform both training and test data
    embeddings_train_reduced = pca_final.transform(embeddings_train)
    embeddings_test_reduced = pca_final.transform(embeddings_test)
    
    return embeddings_train_reduced, embeddings_test_reduced, pca_final

def process_dataset(df, batch_size=64):
    try:
        model, batch_converter = load_esm_model()
        sequences = df['sequence'].values
        y = df['mean_pH'].values
        embeddings = get_esm_embeddings_batched(sequences, model, batch_converter, max_tokens_per_batch=batch_size*1000)
        
        # Perform PCA reduction
        embeddings_reduced, _, pca_model = PCA_reduction(embeddings, embeddings)
        
        return embeddings_reduced, y, pca_model
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

def main(input_csv, seq_col, target_col, output_emb):

    df = pd.read_csv(input_csv)
    sequences = df[seq_col].values.tolist()
    targets = df[target_col].values.tolist()

    model, batch_converter = load_esm_model()

    start = time()
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    finish = time()

    print(f"Time elapsed: {finish - start}")

    X = np.zeros(len(embeddings),
                 dtype=[('X', 'f4', (1280, )), ('y', 'f4')])
    X['X'] = embeddings
    X['y'] = targets
    np.save(output_emb, X)

if __name__ == "__main__":
    fire.Fire(main)
