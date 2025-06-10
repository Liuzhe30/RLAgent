import pubchempy as pcp
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def is_missing_fingerprint(x):
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip().lower() in ['', 'nan', 'none']
    return False
df_all = pd.read_csv('../datasets/middlefile/pos_neg_samples.csv')
'''
  pdb_chain                                 region rna_seq ligand  label
0    1am0_A                           H1 (G,G) GAA       G    AMP      1
1    1am0_A                          H2 (A,G) AACU       A    AMP      1
2    1am0_A  M1 (G,C)  (G,G)  (A,G)  (U,A) G (C,G)       G    AMP      1
3    1arj_N                        H1 (C,G) CUGGGA       C    ARG      1
4    1arj_N                     B1 (A,U) UCU (G,C)       A    ARG      1
'''

unique_ligands = df_all['ligand'].dropna().unique()

output_dir = '../datasets/middlefile/'
round_template = os.path.join(output_dir, 'ligand_pubchem_properties_round{}.csv')
merged_output_path = os.path.join(output_dir, 'ligand_pubchem_properties_merged.csv')
final_merged_with_data = os.path.join(output_dir, 'pos_neg_samples_feature.csv')
retrieved_df = pd.DataFrame()

max_rounds = 3
for round_id in range(1, max_rounds + 1):
    print(f"\nStarting round {round_id}...")

    # Ligands to query this round
    if round_id == 1:
        ligands_to_query = unique_ligands
    else:
        retrieved_df['cactvs_fingerprint'] = retrieved_df['cactvs_fingerprint'].replace(['', None, 'nan', 'NaN'], np.nan)
        lligands_to_query = retrieved_df[
    retrieved_df['cactvs_fingerprint'].isna() | 
    (retrieved_df['cactvs_fingerprint'] == '') |
    (retrieved_df['cactvs_fingerprint'].isnull())
    ]['ligand'].unique()

    print(f"Ligands to query this round: {len(ligands_to_query)}")

    # This round's results
    ligand_info_df = pd.DataFrame(columns=[
        'ligand', 'smile', 'molecular_weight', 'molecular_formula',
        'atom', 'fingerprint', 'cactvs_fingerprint'
    ])

    for ligand in tqdm(ligands_to_query, desc=f"Round {round_id}"):
        try:
            compound = pcp.get_compounds(ligand, 'name')
            if not compound:
                continue
            compound = compound[0]
        except:
            continue

        try:
            smile = compound.isomeric_smiles
        except AttributeError:
            smile = np.nan
        try:
            molecular_weight = compound.molecular_weight
        except AttributeError:
            molecular_weight = np.nan
        try:
            molecular_formula = compound.molecular_formula
        except AttributeError:
            molecular_formula = np.nan
        try:
            atom = compound.atoms
        except AttributeError:
            atom = np.nan
        try:
            fingerprint = compound.fingerprint
        except AttributeError:
            fingerprint = np.nan
        try:
            cactvs_fingerprint = compound.cactvs_fingerprint
        except AttributeError:
            cactvs_fingerprint = np.nan

        ligand_info_df = ligand_info_df.append([{
            'ligand': ligand,
            'smile': smile,
            'molecular_weight': molecular_weight,
            'molecular_formula': molecular_formula,
            'atom': atom,
            'fingerprint': fingerprint,
            'cactvs_fingerprint': cactvs_fingerprint
        }], ignore_index=True)

    # Save this round
    ligand_info_df.to_csv(round_template.format(round_id), index=False)
    print(f"Round {round_id} results saved to: {round_template.format(round_id)}")

    # Merge into master ligand property table
    retrieved_df = pd.concat([retrieved_df, ligand_info_df], ignore_index=True)
    retrieved_df.drop_duplicates(subset='ligand', keep='last', inplace=True)

    # Save merged ligand property table
    retrieved_df.to_csv(merged_output_path, index=False)
    print(f"Merged ligand properties saved to: {merged_output_path}")

    # Early stop if all ligands retrieved
    if retrieved_df['cactvs_fingerprint'].notna().sum() == len(unique_ligands):
        print("All ligands successfully retrieved.")
        break

# Merge back into original dataframe
df_all = df_all.merge(ligand_info_df, on='ligand', how='left')
df_all.to_csv(final_merged_with_data,index=False)
print(df_all.head())
'''
   Unnamed: 0 pdb_chain                                 region rna_seq ligand  \
0           0    1am0_A                           H1 (G,G) GAA       G    AMP   
1           1    1am0_A                          H2 (A,G) AACU       A    AMP   
2           2    1am0_A  M1 (G,C)  (G,G)  (A,G)  (U,A) G (C,G)       G    AMP   
3           3    1arj_N                        H1 (C,G) CUGGGA       C    ARG   
4           4    1arj_N                     B1 (A,U) UCU (G,C)       A    ARG   

   label                                              smile molecular_weight  \ 
0      1  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...           347.22    
1      1  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...           347.22    
2      1  C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H...           347.22    
3      1                        C(C[C@@H](C(=O)O)N)CN=C(N)N           174.20    
4      1                        C(C[C@@H](C(=O)O)N)CN=C(N)N           174.20    

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


import pubchempy as pcp
import pandas as pd
import numpy as np
from tqdm import tqdm

# List of 12 unmatched ligands
unmatched_ligands = ['38E', '747', 'BTN', 'FFO', 'GNG', 'HPA', 'OHX', 'PRP', 'R14', 'ROS', 'TPP', 'TOC']

# Create a dataframe to collect results
supplemental_df = pd.DataFrame(columns=[
    'ligand', 'smile', 'molecular_weight', 'molecular_formula',
    'atom', 'fingerprint', 'cactvs_fingerprint'
])

# Query PubChem
for ligand in tqdm(unmatched_ligands, desc="Querying unmatched ligands from PubChem"):
    try:
        compound = pcp.get_compounds(ligand, 'name')
        if not compound:
            continue
        compound = compound[0]
    except:
        continue

    try:
        smile = compound.isomeric_smiles
    except AttributeError:
        smile = np.nan
    try:
        molecular_weight = compound.molecular_weight
    except AttributeError:
        molecular_weight = np.nan
    try:
        molecular_formula = compound.molecular_formula
    except AttributeError:
        molecular_formula = np.nan
    try:
        atom = compound.atoms
    except AttributeError:
        atom = np.nan
    try:
        fingerprint = compound.fingerprint
    except AttributeError:
        fingerprint = np.nan
    try:
        cactvs_fingerprint = compound.cactvs_fingerprint
    except AttributeError:
        cactvs_fingerprint = np.nan

    supplemental_df = supplemental_df.append([{
        'ligand': ligand,
        'smile': smile,
        'molecular_weight': molecular_weight,
        'molecular_formula': molecular_formula,
        'atom': atom,
        'fingerprint': fingerprint,
        'cactvs_fingerprint': cactvs_fingerprint
    }], ignore_index=True)

print(supplemental_df) # null