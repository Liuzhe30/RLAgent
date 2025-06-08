import pandas as pd
import re
import random

# Step 1: Load the input data file
with open("../datasets/raw/All_site2motif2_new_pdb.txt", "r") as file:
    lines = file.readlines()

# Step 2: Parse each line into structured positive samples
entries = []
current_pdb = None  # Keep track of the current PDB chain ID

for line in lines:
    line = line.strip()
    if not line:
        continue

    # Update current PDB chain ID when a new structure line is found
    if re.match(r'^\w{4}_\w$', line):
        current_pdb = line
        continue

    if current_pdb is None:
        continue

    # Attempt to split RNA region info and ligand
    try:
        region_info, ligand = line.rsplit('\t', 1)
    except ValueError:
        continue  # Skip malformed lines

    # Extract a simplified RNA sequence (first valid A/U/G/C stretch)
    rna_seq_match = re.search(r'[AUGC]+', region_info.replace(' ', ''))
    if rna_seq_match:
        rna_seq = rna_seq_match.group(0)
        entries.append({
            'pdb_chain': current_pdb,
            'region': region_info.strip(),
            'rna_seq': rna_seq,
            'ligand': ligand,
            'label': 1  # Positive sample
        })

# Step 3: Create a DataFrame of positive samples
df_pos = pd.DataFrame(entries)

# Step 4: Build negative samples by randomly pairing RNA with a different ligand
unique_ligands = df_pos['ligand'].unique()
neg_samples_diverse = []

for _, row in df_pos.iterrows():
    other_ligands = [l for l in unique_ligands if l != row['ligand']]
    if other_ligands:
        # Randomly select a different ligand for negative pairing
        neg_ligand = random.choice(other_ligands)
        neg_samples_diverse.append({
            'pdb_chain': row['pdb_chain'],
            'region': row['region'],
            'rna_seq': row['rna_seq'],
            'ligand': neg_ligand,
            'label': 0  # Negative sample
        })

# Step 5: Combine positive and negative samples
df_all = pd.concat([df_pos, pd.DataFrame(neg_samples_diverse)], ignore_index=True)
print(df_all.head())

df_all.to_csv('../datasets/middlefile/pos_neg_samples.csv')
