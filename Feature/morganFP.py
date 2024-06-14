import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 1024

def extract_and_save_morgan_fingerprints(input_smiles_file, output_csv_file):
    data = pd.read_csv(input_smiles_file)

    fingerprint_df = pd.DataFrame()

    for index, row in data.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:

            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NUM_BITS)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)


            fingerprint_df[f'feature_{index}'] = features


    fingerprint_df = fingerprint_df.T


    fingerprint_df.to_csv(output_csv_file, header=False, index=False)


input_smiles_file = r'D:\Coding\P1\ECAmyloid-main\ECAmyloid-main\dataset\test_dataset_smiles.csv'
output_csv_file = r'D:\Coding\P1\ECAmyloid-main\ECAmyloid-main\dataset\test_dataset_morgan.csv'

extract_and_save_morgan_fingerprints(input_smiles_file, output_csv_file)

