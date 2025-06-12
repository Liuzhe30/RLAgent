import numpy as np
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import re
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('../datasets/middlefile/pos_neg_samples_ligand.csv')
#print(df.head())
'''
  pdb_chain                                 region rna_seq ligand  label  \
0    1am0_A                           H1 (G,G) GAA       G    AMP      1
1    1am0_A                          H2 (A,G) AACU       A    AMP      1
2    1am0_A  M1 (G,C)  (G,G)  (A,G)  (U,A) G (C,G)       G    AMP      1
3    1arj_N                        H1 (C,G) CUGGGA       C    ARG      1
4    1arj_N                     B1 (A,U) UCU (G,C)       A    ARG      1

                                               smile  molecular_weight  \
0  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
1  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
2  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
3                        C(C[C@@H](C(=O)O)N)CN=C(N)N            174.20
4                        C(C[C@@H](C(=O)O)N)CN=C(N)N            174.20

  molecular_formula                                               atom  \
0       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
1       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
2       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
3         C6H14N4O2  [Atom(1, O), Atom(2, O), Atom(3, N), Atom(4, N...
4         C6H14N4O2  [Atom(1, O), Atom(2, O), Atom(3, N), Atom(4, N...

                                         fingerprint  \
0  00000371C073B802000000000000000000000000000162...
1  00000371C073B802000000000000000000000000000162...
2  00000371C073B802000000000000000000000000000162...
3  00000371C063B000000000000000000000000000000000...
4  00000371C063B000000000000000000000000000000000...

                                  cactvs_fingerprint
0  1100000001110011101110000000001000000000000000...
1  1100000001110011101110000000001000000000000000...
2  1100000001110011101110000000001000000000000000...
3  1100000001100011101100000000000000000000000000...
4  1100000001100011101100000000000000000000000000...
'''

df[['pdb_id', 'chain_id']] = df['pdb_chain'].str.extract(r'([0-9a-zA-Z]{4})_([A-Za-z0-9])')

def fetch_rna_sequence_from_rcsb(pdb_id, chain_id):

    try:
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        fasta_text = soup.get_text()
        entries = fasta_text.strip().split('>')

        for entry in entries:
            if not entry:
                print(pdb_id, chain_id)
                continue
            header, *sequence_lines = entry.strip().split('\n')

            #if f"{chain_id.upper()}" in header and "RNA" in header.upper():
            if f"{chain_id.upper()}" in header:
                sequence = ''.join(sequence_lines).replace(" ", "").strip()
                print(sequence)
                return sequence
                
        return None
    except Exception as e:
        return None

df['rna_sequence'] = df.apply(lambda row:fetch_rna_sequence_from_rcsb(row['pdb_id'], row['chain_id']), axis=1)
print(df.head())
'''
  pdb_chain                                 region rna_seq ligand  label  \
0    1am0_A                           H1 (G,G) GAA       G    AMP      1
1    1am0_A                          H2 (A,G) AACU       A    AMP      1
2    1am0_A  M1 (G,C)  (G,G)  (A,G)  (U,A) G (C,G)       G    AMP      1
3    1arj_N                        H1 (C,G) CUGGGA       C    ARG      1
4    1arj_N                     B1 (A,U) UCU (G,C)       A    ARG      1

                                               smile  molecular_weight  \
0  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
1  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
2  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...            347.22
3                        C(C[C@@H](C(=O)O)N)CN=C(N)N            174.20
4                        C(C[C@@H](C(=O)O)N)CN=C(N)N            174.20

  molecular_formula                                               atom  \
0       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
1       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
2       C10H14N5O7P  [Atom(1, P), Atom(2, O), Atom(3, O), Atom(4, O...
3         C6H14N4O2  [Atom(1, O), Atom(2, O), Atom(3, N), Atom(4, N...
4         C6H14N4O2  [Atom(1, O), Atom(2, O), Atom(3, N), Atom(4, N...

                                         fingerprint  \
0  00000371C073B802000000000000000000000000000162...
1  00000371C073B802000000000000000000000000000162...
2  00000371C073B802000000000000000000000000000162...
3  00000371C063B000000000000000000000000000000000...
4  00000371C063B000000000000000000000000000000000...

                                  cactvs_fingerprint pdb_id chain_id  \
0  1100000001110011101110000000001000000000000000...   1am0        A
1  1100000001110011101110000000001000000000000000...   1am0        A
2  1100000001110011101110000000001000000000000000...   1am0        A
3  1100000001100011101100000000000000000000000000...   1arj        N
4  1100000001100011101100000000000000000000000000...   1arj        N

                               rna_sequence
0  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC
1  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC
2  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC
3             GGCAGAUCUGAGCCUGGGAGCUCUCUGCC
4             GGCAGAUCUGAGCCUGGGAGCUCUCUGCC
'''
df.to_csv("../datasets/middlefile/pos_neg_samples_ligand_rna.csv", index=False)