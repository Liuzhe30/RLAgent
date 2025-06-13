import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Load the input file
df = pd.read_csv("../datasets/middlefile/pos_neg_samples_ligand_rna.csv")

# Step 1: Region parser using finditer to retain order and type
def parse_region_with_pairs(region_str):
    structure_type = None
    region_pairs = []

    if pd.isna(region_str):
        return structure_type, region_pairs

    structure_match = re.match(r"([A-Z]+[0-9]*)", region_str)
    if structure_match:
        structure_type = structure_match.group(1)

    tokens = re.finditer(r"\(([AUCG]),([AUCG])\)|\b[AUCG]{1,}\b", region_str)
    current_pairs = []

    for match in tokens:
        if match.group(1) and match.group(2):  # basepair match
            current_pairs.append((match.group(1), match.group(2)))
        else:  # region sequence
            region_seq = match.group(0)
            region_pairs.append((region_seq, current_pairs))
            current_pairs = []

    return structure_type, region_pairs

# Step 2: Apply region decomposition and RNA sequence alignment
expanded_rows = []

for _, row in df.iterrows():
    structure_type, region_components = parse_region_with_pairs(row['region'])

    if not region_components:
        continue

    for region_seq, basepairs in region_components:
        rna_seq = row['rna_sequence']
        if isinstance(rna_seq, str) and region_seq in rna_seq:
            start = rna_seq.find(region_seq)
            end = start + len(region_seq)
            mask = [1 if start <= i < end else 0 for i in range(len(rna_seq))]
        else:
            continue

        expanded_rows.append({
            **row,
            'structure_type': structure_type,
            'region_seq': region_seq,
            'basepairs': basepairs,
            'region_start': start,
            'region_end': end,
            'region_mask': mask
        })

# Step 3: Export result
expanded_df = pd.DataFrame(expanded_rows)
print(expanded_df.head())
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

                               rna_sequence structure_type region_seq  \
0  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC             H1        GAA
1  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC             H2       AACU
2  GGGUUGGGAAGAAACUGUGGCACUUCGGUGCCAGCAACCC             M1          G
3             GGCAGAUCUGAGCCUGGGAGCUCUCUGCC             H1     CUGGGA
4             GGCAGAUCUGAGCCUGGGAGCUCUCUGCC             B1        UCU

                          basepairs  region_start  region_end  \
0                          [(G, G)]             7          10
1                          [(A, G)]            12          16
2  [(G, C), (G, G), (A, G), (U, A)]             0           1
3                          [(C, G)]            13          19
4                          [(A, U)]             6           9

                                         region_mask
0  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, ...
1  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...
2  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
3  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ...
4  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...
'''

# Step 4: Save to file
output_path = "../datasets/dataset_annotation.csv"
expanded_df.to_csv(output_path, index=False)

