import numpy as np
import pandas as pd
import os


# List of 12 unmatched ligands
unmatched_ligands = ['38E', '747', 'BTN', 'FFO', 'GNG', 'HPA', 'OHX', 'PRP', 'R14', 'ROS', 'TPP', 'TOC']
df = pd.read_csv('../datasets/middlefile/pos_neg_samples_feature.csv')
df = df[~df['ligand'].isin(unmatched_ligands)]
df = df.drop(columns=['Unnamed: 0'])
df = df.reset_index(drop=True)
df.to_csv('../datasets/middlefile/pos_neg_samples_ligand.csv',index=False)