import fm
import torch
import requests
from rdkit import Chem
import numpy as np
from scipy import sparse
from transformers import BertTokenizer, BertModel

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
    return results["representations"][12][0].cpu().numpy()


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
    fingerprint = [int(fingerprint[i], 16) for i in range(0, len(fingerprint))]
    return np.array(fingerprint)

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
    return np.array(splitted_smiles)

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
    return np.array(one_hot_coding(ligand, words, max_len=len(ligand)).toarray().tolist())

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

from rlagent.llm_tools import run_LLM_json_auto_retry

def TaskReconginzeAgent(input_text):
    # dialogue_text = "\n".join([f"用户: {u}\nAI: {a}" for u, a in conversation_history])
    prompt = (
        "You are an expert in the field of RNA molecular structure and function. Please break down the user's question based on their inquiry："
        f"The user's question is as follows:\n{input_text}\n"
    )
    prompt += (
        "\nPlease provide a preliminary assessment and return the content in JSON format, including the following fields:\n"
        "- clear_enough: The user's description of the question is clear enough to be rated as 1, otherwise it should be rated as 0, with no other possibilities.\n"
        "- using_protein_name: The RNA the user intends to inquire about is referred to by its name or identifier, rated as 1; if presented as a nucleotide sequence, it is rated as 0. Any other situation is categorized as the user's description being unclear.\n"
        "- protein: The RNA name or nucleotide sequence that the user intends to inquire about.\n"
        "- question: The user's question description should refer to the target RNA as 'this RNA', for example, 'What diseases is this RNA associated with?'"
    )
    return run_LLM_json_auto_retry(prompt)

import fm
import torch
# 1. Load RNA-FM model
device = 'cuda:2'
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
batch_labels, batch_strs, batch_tokens = batch_converter([('whatever', 'TTACGGA'.replace('T','U'))])
batch_tokens = batch_tokens.to(device)
model.to(device)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
results["representations"][12].mean(dim = 1)[0].tolist()
import os
import json
folder_path = '/data/lyk/shijie/RNA_LLM/database/mirnas'  
dict_sum = {}
# 获取文件夹下所有文件名
for filename in os.listdir(folder_path):
    # if os.path.isfile(os.path.join(folder_path, filename)):
    with open(os.path.join(folder_path, filename), 'r') as file:
        data = json.load(file)
        # print(len(data), filename.split('.')[0])
    dict_sum[filename.split('.')[0]] = data

import pandas as pd
import math
class SequenceReconginzeAgent:
    def __init__(self, dict_sum):
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(device).eval()
        self.data = dict_sum
        self.rna_information = pd.read_csv('/data/lyk/shijie/RNA_LLM/database/all_mirna.csv')
    def analyze_conversation(
        self,
        sequence
    ):

        # last_user_input = conversation_history[-1][0] if conversation_history else ""
        match_protein_id, similarity = self.protein_match(sequence)
        # 构造诊断分析 prompt
        analysis_prompt = self._build_analysis_prompt(match_protein_id, similarity)

        # 请求 LLM 获取诊断分析结果
        # print(llm_response)
        # print(self.extract_json_from_text(llm_response))
        # 尝试解析 JSON 格式的诊断结果
        # analysis = self._parse_llm_response(llm_response)

        # if "diagnosis_suspect" in analysis:
        #     return {
        #         "type": "start_assessment_query",
        #         "diagnosis_suspect": analysis["diagnosis_suspect"],
        #         "content": analysis["content"]
        #     }

        # 否则继续普通对话生成
        # response_prompt = self._build_response_prompt(last_user_input)
        # full_response = await self.llm.generate(response_prompt)

        # thought, actual_response = self._extract_thought_response(full_response)
        # return llm_response
        answer = run_LLM_json_auto_retry(analysis_prompt)
        answer['protein_name'] = match_protein_id
        return answer
        # return {
        #     "type": "dialogue",
        #     "thought": thought,
        #     "content": actual_response
        # }

    def _build_analysis_prompt(
        self,
        RNA_name,
        similarity
    ) -> str:
        # dialogue_text = "\n".join([f"用户: {u}\nAI: {a}" for u, a in conversation_history])
        prompt = (
            "You are an expert in the field of RNA molecular structure and function. Previously, you matched an unfamiliar nucleotide sequence with the most similar reference RNA. Please assess the reliability of the reference RNA based on the reference RNA name, reference RNA information, and the cosine similarity between the feature vectors of the model RNA and the reference RNA:"
            f"Reference protein name.：\n{RNA_name}\n"
            f"Cosine similarity：\n{similarity}\n"
        )
        prompt += (
            "\nPlease provide a preliminary assessment and return the content in JSON format, including the following fields:\n"
            "- Reliable: Based on the overall similarity assessment, if the similarity is sufficient, classify the reference RNA as a reliable similar RNA for the model RNA with a value of 1; otherwise, assign a value of 0, with no other conditions.\n"
            "- reason: Reason for acceptance or rejection.\n"
        )
        return prompt
    def is_valid_braces(self, s):
    # 检查是否只包含一个 '{' 和一个 '}'
        return s.count('{') == 1 and s.count('}') == 1 and s.index('{') < s.index('}')
    
    def protein_match(self, sequence):
        max_similarity = 0
        batch_labels, batch_strs, batch_tokens = self.batch_converter([('whatever', sequence.replace('T','U'))])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token = results["representations"][12].mean(dim = 1)[0].tolist()
        for key,value in self.data.items():
            similarity = self.cosine_similarity(value, token)
            if similarity > max_similarity:
                max_similarity = similarity
                protein_id = key
        name = self.rna_information[self.rna_information['id'] == protein_id].iloc[0]['name']
        return name, max_similarity


    def cosine_similarity(self, list1, list2):
        dot_product = sum(a * b for a, b in zip(list1, list2))
        norm1 = math.sqrt(sum(a ** 2 for a in list1))
        norm2 = math.sqrt(sum(b ** 2 for b in list2))
        return dot_product / (norm1 * norm2)
    def query_database(self, protein_name):
        answer = {}
        if self.protein_information['preferred_name'].str.contains(protein_name).any():
            return {'Reliable': 1 ,'protein_name': protein_name ,'protein_annotation': self.protein_information[self.protein_information['preferred_name'] == protein_name].iloc[0]['annotation'], 'reason': ''}
        else:
            return {'Reliable': 0 ,'protein_name': protein_name ,'protein_annotation': '' , 'reason': "I'm sorry, but the target RNA could not be found in our knowledge base, preventing the generation of related information. Please check your input or try submitting the RNA sequence in a different format."}





from typing import List, Tuple, Dict, Optional
import json
import re
class KnowledgeGraphAgent:
    def __init__(self):

        self.nodes = pd.read_csv('/data/lyk/shijie/RNA_LLM/database/rnakg/selected_node.csv', index_col = 0)
        self.edge = pd.read_csv('/data/lyk/shijie/RNA_LLM/database/rnakg/edges_selected.csv', index_col = 0)
        # self.circrna_disease = pd.read_csv('/data/lyk/shijie/RNA_LLM/database/KGANCDA-main/Dataset/Dataset2/circrna-disease.txt', sep = ',', header = None).head()
        # self.circrna_mirna = pd.read_csv('/data/lyk/shijie/RNA_LLM/database/KGANCDA-main/Dataset/Dataset2/circrna-mirna.txt', sep = ',', header = None).head()
    def analyze_conversation(
        self,
        RNA_name
    ):
        """完全基于LLM的对话分析逻辑"""

        # last_user_input = conversation_history[-1][0] if conversation_history else ""
        relations = self.get_relations(RNA_name)
        # 构造诊断分析 prompt
        analysis_prompt = self._build_analysis_prompt(RNA_name, relations)


        # print(llm_response)
        # print(self.extract_json_from_text(llm_response))
        # 尝试解析 JSON 格式的诊断结果
        # analysis = self._parse_llm_response(llm_response)

        # if "diagnosis_suspect" in analysis:
        #     return {
        #         "type": "start_assessment_query",
        #         "diagnosis_suspect": analysis["diagnosis_suspect"],
        #         "content": analysis["content"]
        #     }

        # 否则继续普通对话生成
        # response_prompt = self._build_response_prompt(last_user_input)
        # full_response = await self.llm.generate(response_prompt)

        # thought, actual_response = self._extract_thought_response(full_response)
        # return llm_response
        return run_LLM_json_auto_retry(analysis_prompt)
        # return {
        #     "type": "dialogue",
        #     "thought": thought,
        #     "content": actual_response
        # }

    def _build_analysis_prompt(
        self,
        RNA_name,
        relations
    ) -> str:
        # dialogue_text = "\n".join([f"用户: {u}\nAI: {a}" for u, a in conversation_history])
        prompt = (
            "You are an expert in the field of RNA molecular structure and function. Please generate a textual report based on the RNA name and its associations with other elements："
            f"The RNA name is as follows:：\n{RNA_name}\n"
            f"The associated elements are as follows:"
        )
        for one_relation in relations:
            prompt +=(one_relation + '\n')
        prompt += (
            "\nPlease provide a preliminary assessment and return the content in JSON format, including the following fields:\n"
            "- name: The RNA name\n"
            "- text: The report for this RNA, do not \n"
        )
        return prompt
    def extract_json_from_text(self, text):
        try:
            # 匹配 <think>...</think> 之外的所有内容（包括换行）
            pattern = r'(?:<think>.*?</think>)|(?P<content>[^<]+)'
            matches = re.finditer(pattern, text, re.DOTALL)
            
            # 拼接所有非 <think> 的内容
            non_think_parts = [
                match.group("content") 
                for match in matches 
                if match.group("content")
            ]
            non_think_text = "".join(non_think_parts).strip()
            
            # 如果内容是 JSON，尝试解析
            # if non_think_text.startswith("{") and non_think_text.endswith("}"):
            if self.is_valid_braces(non_think_text):
                return json.loads('{' + non_think_text.split('{')[1].split('}')[0] + '}')
            return non_think_text if non_think_text else None

        except (json.JSONDecodeError, AttributeError):
            return None
    def is_valid_braces(self, s):
    # 检查是否只包含一个 '{' 和一个 '}'
        return s.count('{') == 1 and s.count('}') == 1 and s.index('{') < s.index('}')
    def get_relations(self, rna_name):
        node_name = self.nodes[self.nodes['describ'] == rna_name].iloc[0]['name']
        one_jumps = self.edge[(self.edge['subject'] == node_name) | (self.edge['object'] == node_name)]
        relations = []
        for i in range(len(one_jumps)):
            use = one_jumps.iloc[i]
            information = self.nodes[self.nodes['name'] == use['subject']].iloc[0]
            discrib_a = information['type'] + ':"' + information['describ'] + '"'
            information = self.nodes[self.nodes['name'] == use['object']].iloc[0]
            discrib_b = information['type'] + ':"' + information['describ'] + '"'
            relations.append(discrib_a + ' ' + use['predicate'] + ' ' + discrib_b)
        return relations



class AnswerAgent:

    def analyze_conversation(
        self,
        question,
        protein_name,
        protein_annotation
    ):


        # last_user_input = conversation_history[-1][0] if conversation_history else ""


        # 构造诊断分析 prompt
        analysis_prompt = self._build_analysis_prompt(question, protein_name, protein_annotation)

        llm_response = run_LLM(analysis_prompt)
        # print(llm_response)
        # 尝试解析 JSON 格式的诊断结果
        # analysis = self._parse_llm_response(llm_response)

        # if "diagnosis_suspect" in analysis:
        #     return {
        #         "type": "start_assessment_query",
        #         "diagnosis_suspect": analysis["diagnosis_suspect"],
        #         "content": analysis["content"]
        #     }

        # 否则继续普通对话生成
        # response_prompt = self._build_response_prompt(last_user_input)
        # full_response = await self.llm.generate(response_prompt)

        # thought, actual_response = self._extract_thought_response(full_response)
        # return llm_response
        # return self.extract_json_from_text(llm_response)
        return llm_response


    def _build_analysis_prompt(
        self,
        question,
        rna_name,
        rna_annotation
    ) -> str:
        # dialogue_text = "\n".join([f"用户: {u}\nAI: {a}" for u, a in conversation_history])
        prompt = (
            "You are an expert in the field of RNA molecular structure and function. Please provide answers based on the user's questions and any additional information they provide:"
            f"The user's question is as follows:\n{question}\n"
            f"The user wants to inquire about the RNA:\n{rna_name}\n"
            f"The additional information is as follows:\n{rna_annotation}\n"
            "Please do not provide a response that exceeds 1500 words."
        )
        # prompt += (
        #     "\n请给出一个初步判断，并返回 JSON 格式内容，包括以下字段：\n"
        #     "- clear_enough: 用户对问题的描述足够清晰为1，否则为0,无其他情况\n"
        #     "- using_protein_name: 用户要提问的蛋白质是以蛋白名或基因名形式提出的为1，以氨基酸序列形式突出为0，其他情况归于用户描述不够清晰\n"
        #     "- protein: 用户要提问的蛋白名或氨基酸序列\n"
        #     "- question: 用户的问题描述，目标蛋白请用‘该蛋白’指代，如‘查询该蛋白能构成什么蛋白复合物’"
        # )
        return prompt


class Supervisor:

    def __init__(self):
        # self.task_reconginze_agent = TaskReconginzeAgent()
        self.sequence_reconginze_agent = SequenceReconginzeAgent(dict_sum)
        self.KGagent = KnowledgeGraphAgent()
        self.answerAgent = AnswerAgent()
        self.device = 'cuda:2'
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertModel.from_pretrained("bert-large-uncased")
        self.model.to(self.device)
    def process_input(self, user_input: str):
        # session = self._get_session('default')
        # colored_input = f"{self.BLUE}{user_input}{self.RESET}"        
        
        # 记录用户输入
        # session["conversation"].append((colored_input, None))
        
        # 格式化响应
        task = TaskReconginzeAgent(user_input)
        if task['using_protein_name'] == 0:
            protein_reconginze_informations = self.sequence_reconginze_agent.analyze_conversation(task['protein'])

        else:#查询蛋白名称
            protein_reconginze_informations = self.sequence_reconginze_agent.query_database(task['protein'])
        if protein_reconginze_informations['Reliable'] == 0: #输入和所有蛋白都不匹配
            return protein_reconginze_informations['reason']
        else:
            protein_name = protein_reconginze_informations['protein_name']  
        addition_information = self.KGagent.analyze_conversation(protein_name)
        answer = self.answerAgent.analyze_conversation(task['question'], protein_name, addition_information['text'])
        return answer

    def input2token(self, user_input: str):
        # session = self._get_session('default')
        # colored_input = f"{self.BLUE}{user_input}{self.RESET}"        
        
        # 记录用户输入
        # session["conversation"].append((colored_input, None))
        
        # 格式化响应
        task = TaskReconginzeAgent(user_input)
        if task['using_protein_name'] == 0:
            protein_reconginze_informations = self.sequence_reconginze_agent.analyze_conversation(task['protein'])

        else:#查询蛋白名称
            protein_reconginze_informations = self.sequence_reconginze_agent.query_database(task['protein'])
        if protein_reconginze_informations['Reliable'] == 0: #输入和所有蛋白都不匹配
            return protein_reconginze_informations['reason']
        else:
            protein_name = protein_reconginze_informations['protein_name']  
        addition_information = self.KGagent.analyze_conversation(protein_name)
        answer = self.answerAgent.analyze_conversation(task['question'], protein_name, addition_information['text'])
        encoded_input = {q: k.to(self.device) for q,k in tokenizer(answer, return_tensors='pt').items()}
        return model(**encoded_input)

agent = Supervisor()
def sequence2text2token(sequence, agent = agent):
    answer = agent.input2token(f'introduce the function of {sequence} please')
    return answer['last_hidden_state'].squeeze(0).detach().cpu().numpy()

def sequence2text2vector(sequence, agent = agent):
    answer = agent.input2token(f'introduce the function of {sequence} please')
    return answer['pooler_output'].squeeze(0).detach().cpu().numpy()
