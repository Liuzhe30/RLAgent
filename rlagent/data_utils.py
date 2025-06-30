# rlagent/data_utils.py

import pandas as pd
# from rlagent.llm_tools import feature_recongnization_agent
import requests
import torch
import fm
import json
import json
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from scipy import sparse
import os
import json
from tqdm import tqdm
import fm
import torch
# 1. Load RNA-FM model
device = 'cuda:2'
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
def process_RNA(sequence):
    batch_labels, batch_strs, batch_tokens = batch_converter([('whatever', sequence)])
    batch_tokens = batch_tokens.to(device)
    model.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    return results["representations"][12][0].tolist()
def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)

        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return splitted_smiles

def get_maxlen(all_smiles, kekuleSmiles=True):
    maxlen = 0
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        maxlen = max(maxlen, len(spt))
    return maxlen

def get_dict(all_smiles, kekuleSmiles=True):
    words = [' ']
    for smi in all_smiles:
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)

    return words

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j,w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len, len(words)))
    return output


def ligand2onehot(ligand, words):
    return one_hot_coding(ligand, words, max_len=len(ligand)).toarray().tolist()

def str2list(text):
    return json.loads(text)


def get_fingerprint(smile):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + smile + "/JSON"
    
    # 发送请求
    response = requests.get(url)
    
    # 检查响应
    if response.status_code == 200:
        data = response.json()
        # print(data)  
    else:
        print("Error retrieving data from PubChem. " + smile)
        return '0'* 230
    fingerprint = None
    for compound in data.get("PC_Compounds", []):
        for prop in compound.get("props", []):
            if prop.get("urn", {}).get("label") == "Fingerprint":
                fingerprint = prop.get("value", {}).get("binary")
                break
        if fingerprint:
            break
    return fingerprint

def hex_to_fixed_length_list(hex_string):
    return [int(char, 16) for char in hex_string]

def process_rna_sequence(seq, model, batch_converter, device):
    batch_labels, batch_strs, batch_tokens = batch_converter([('hello', seq)])
    batch_tokens = batch_tokens.to(device)
    results = model(batch_tokens, repr_layers=[12])
    return results["representations"][12].mean(dim = 1).tolist()[0]


def concatenate_lists(row):
    return row['coding_1'] + row['coding_2']


def check_and_process_data(input_path, output_path, element_use, using_RAG):
    try:
        data = pd.read_csv(input_path)

        required_columns = {"ligand", "label", "rna_sequence", "region_mask"}

        if not required_columns.issubset(data.columns):
            print(f"Missing required columns! Expected: {required_columns}, Found: {list(data.columns)}")
            return None

        # Example processing (you can add more if needed)
        # For now, just save the loaded data to processed file
        # Add onehot to every raws
        # for column in element_use:
        #     data = feature_recongnization(column, data)
        # if using_RAG:
        #     data = add_RAG(column, data)
        # Add embedding to every ligand
        # if 'RAG' in element_use:
            # add RAG message to every raws



        # Base process
        # data['fingerprint'] = data['ligand'].apply(get_fingerprint)
        # data['coding_1'] = data['fingerprint'].apply(hex_to_fixed_length_list)
        # # data['codeing_2'] = data['rna_sequence'].apply(hex_to_fixed_length_list)
        # model, alphabet = fm.pretrained.rna_fm_t12()
        # batch_converter = alphabet.get_batch_converter()
        # device = 'cuda:3'
        # model.eval()
        # model.to(device)
        # data['coding_2'] = data['rna_sequence'].apply(lambda seq: process_rna_sequence(seq, model, batch_converter, device))
        # data['coding'] = data.apply(concatenate_lists, axis=1)



        # if 'pretrained' in element_use:
            # add pretrained to every raws




        # data.to_csv(output_path, index=False)

        data = pd.read_csv(output_path)
        data['coding'] = data['coding'].apply(str2list)

        words = get_dict(data['ligand'].tolist())
        data['ligand_feature'] = data['ligand'].apply(lambda x: ligand2onehot(x, words))
        data['rna_feature'] = data['rna_sequence'].apply(process_RNA)
        data['region_mask'] = data['region_mask'].apply(str2list)
        for index, row in data.iterrows():
            # 获取当前行的 region_mask 和 rna_feature
            mask = [0] + row['region_mask'] + [0]
            features = row['rna_feature']

            # 将 mask 添加到 features 中
            for i in range(len(features)):
                features[i].append(mask[i])

            # 更新当前行的 rna_feature

            data.at[index, 'rna_feature'] = features
        print(f"Loaded {len(data)} rows from {input_path}, saved to {output_path}")
        data.to_csv(output_path, index=False)
        return data

    except Exception as e:
        print(f"Error processing data: {e}")
        return None


# def feature_recongnization(column, pd):
#     ready = False
#     user_input_history = []
#     while ready != True:
#         answer = feature_recongnization_agent(user_input_history, column, pd)
#         user_input_history.append(answer)
#         user_input = input("\n[Your reply] → ").strip()
