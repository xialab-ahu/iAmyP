import numpy as np
import pandas as pd
from collections import Counter
import re


def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = []
    for i in AA:
        header.append(i)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def read_fasta(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    fasta_data = []
    current_sequence = ""
    current_name = ""

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_name and current_sequence:
                fasta_data.append((current_name, current_sequence))
            current_name = line[1:]
            current_sequence = ""
        else:
            current_sequence += line

    if current_name and current_sequence:
        fasta_data.append((current_name, current_sequence))

    return fasta_data
def main():
    fasta_data = read_fasta(r"/aggregating peptides/11-20peptide/11-20aggpeptide.fasta")

    aac_features, aac_header = AAC(fasta_data)
    aac_df = pd.DataFrame(aac_features, columns=aac_header)

    aac_df.to_csv(r"D:\Coding\P1\aggregating peptides\11-20peptide\11-20aggpeptide_AAC.csv", index=False)


if __name__ == "__main__":
    main()