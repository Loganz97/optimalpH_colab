![OptimalpHColab_Colab](https://github.com/user-attachments/assets/fabef7a7-30ab-4245-b355-c235214b0f0b)

# OptimalpH-Colab: Predict Optimal pH for Hydrolase Activity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HE5B2Oy82zmgLyB4lhPOly-JPJKfT4A9?usp=sharing)

## Introduction

OptimalpH is a tool designed to predict the optimal pH range for hydrolase activity using machine learning models. This Google Colab notebook allows for batch processing of protein sequences to predict the pH range where hydrolases exhibit maximum catalytic activity.

The model utilizes ESM-2-650M protein language model embeddings developed by Meta AI Research and trains machine learning models including KNN, XGBoost, and K-mers using embeddings labeled with optimal pH data and organismal optimal growth data.

### Author Information

This notebook is a user-friendly interface for predicting a collection of metrics that aid in the expression, purification, and screening of hydrolases created by **Logan Hessefort** ([LinkedIn](https://www.linkedin.com/in/logan-hessefort/)).

This project was created as part of a **US Department of Energy SCGSR Award** ([details](https://science.osti.gov/wdts/scgsr)) at the National Renewable Energy Laboratory with additional support from the **US National Science Foundation** ([grant](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2132183&HistoricalAwards=false)).

### Training Data

The models were trained using approximately **2,840 data points from the BRENDA** enzyme database and **169,517 data points from the Mean Growth pH dataset**. The data were processed through clustering techniques such as DBSCAN and CD-HIT, followed by hyperparameter tuning and evaluation on various splits to optimize prediction accuracy.

## How to Use

1. Click the "Open in Colab" button at the top of this README.
2. Run the cells sequentially in the notebook.
3. Upload your CSV file when prompted (see Input CSV Structure below).
4. Select the desired model type.
5. The notebook will process each sequence and output a consolidated CSV file with the results.

### Input CSV Structure

Your input CSV should contain the following columns:

- `ID`: A unique identifier for each protein sequence
- `Sequence`: The amino acid sequence of the protein

Example format:

| ID       | Sequence       |
|:--------:|:--------------:|
| Protein1 | MVKPKLFYV...   |
| Protein2 | MGSRHYASS...   |

**Note:** Ensure all sequences are valid amino acid sequences using single-letter codes.

### Output

The script generates a comprehensive CSV file containing:

- ID
- Sequence
- Molecular Weight
- Oxidized Cystine Extinction Coefficient
- Isoelectric Point
- Predicted Optimal pH
- Lysine Count
- Arginine Count
- Cysteine Count
- Instability Index
- Stability Prediction (Stable, Moderately Stable, Unstable)
- Hydrophobicity Score

## Key Protein Properties Explained

- **GRAVY Score**: Measures overall protein hydrophobicity (range typically -2 to +2).
- **Isoelectric Point (pI)**: pH at which a protein carries no net electrical charge (range typically 3 to 12).
- **Instability Index**: Estimates protein stability in vitro (≤40 predicted as stable, >40 as unstable).

## Local Installation
If you prefer to run the tool locally:

1. Clone the repository:
   ```
   git clone https://github.com/Loganz97/optimalpH_colab.git
   ```

2. Create a conda environment:
   ```
   conda create --file environment.yml
   conda activate optimalpH
   ```

3. Run the prediction script:
   ```
   python3 code/predict.py --input_csv sequences.csv --model model_xgboost
   ```

   This will use default settings and save the output in the current directory.

   For more options:
   ```
   python3 code/predict.py --help
   ```

   You can specify custom column names or output file:
   ```
   python3 code/predict.py --input_csv sequences.csv --model model_xgboost --output_csv results.csv --id_col ProteinID --seq_col ProteinSequence
   ```

## Citation and Acknowledgment
This tool is based on the research presented in the paper "Approaching Optimal pH Enzyme Prediction with Large Language Models" by Mark Zaretckii, Pavel Buslaev, Igor Kozlovskii, Alexander Morozov, and Petr Popov. The original paper can be accessed [here](https://doi.org/10.1021/acssynbio.4c00465).

## Issues and Questions
For any issues or questions, please open an issue in this GitHub repository.
