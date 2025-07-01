# rlagent/llm_tools.py

import requests
import httpx
from typing import Optional
import asyncio
import json


def run_LLM(prompt: str) -> str:
    api_key = ""
    api_url = "http://192.168.0.108:1025/v1/chat/completions"
    scales_directory = "scales"
    try:
        response = asyncio.run(client.post(
            api_url,
            json={
                "model": "llama_70b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.7
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
        ))
        response.raise_for_status()
        result = response.json()
        # print(result)
        # client.aclose()
        return result["choices"][0]["message"]["content"].split('</think>')[-1] if response.text else "no useful message return"
    except httpx.RequestError as e:
        print(f"requests false: {str(e)}")
        # client.aclose()
        return "requests false"
    except Exception as e:
        print(f"error: {str(e)}")
        # client.aclose()
        return "system error, please try later."



# def run_LLM(full_prompt: str) -> str:
#     OLLAMA_URL = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "deepseek-v2:16b",
#         "prompt": full_prompt,
#         "stream": False,
#     }

#     try:
#         response = requests.post(OLLAMA_URL, json=payload)
#         response.raise_for_status()
#     except Exception as e:
#         print("Error calling Ollama API:", str(e))
#         return False

#     result = response.json()
#     llm_reply = result.get("response", "").strip()
#     return llm_reply
client = httpx.AsyncClient(timeout=60.0)

def answer2json(string):
    start_index = string.find('{')
    end_index = string.rfind('}') + 1
    json_string = string[start_index:end_index]
    return json.loads(json_string)    

def ask_llm_stage(stage_prompt: str, user_context: str = "") -> str:
    """
    Generate agent reply for current stage.
    """

    full_prompt = f"""
        You are RLAgent, an interactive assistant for RNA-Ligand interaction modeling.
        
        Your task is to guide the user step by step through the modeling pipeline.
        
        Current task: {stage_prompt}
        
        Constraints:
        - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
        - Only provide clear and concise instructions to the user.
        - Do not apologize or explain your capabilities.
        - Wait for user reply before continuing.
        
        Now, explain to the user how to prepare their training dataset, using the following information:
        
        ---
        
        You need to prepare a dataset in CSV format.
        
        Each row should contain:
        - ligand: Ligand name
        - label: 1 (positive) or 0 (negative), indicating interaction
        - rna_sequence: RNA sequence (string of A/U/G/C)
        - region_mask: A list of 0/1 values of the same length as RNA sequence, where 1 indicates the region to be predicted
        
        Here is an example:
        
        | ligand | label | rna_sequence       | region_mask                            |
        |--------|-------|--------------------|----------------------------------------|
        | CIR    | 1     | ACGGUUAGGUCGCU     | [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] |
        
        Once your data is ready, please save it to:
        `datasets/train.csv`  
        with the correct format as shown above.
        
        ---
        
        After presenting this to the user, do not output any additional content. Wait for the user to reply.
        Please prompt the user that if the data is ready, Please press yes when data is ready...
        
        User last reply:
        {user_context}
        
        Now generate your reply:
        """

    llm_message = run_LLM(full_prompt)

    print("\n--- Agent says ---")
    print(llm_message)

    return llm_message




def judge_user_ready(user_reply: str) -> bool:
    """
    Use LLM to judge whether the user's reply indicates that they are ready.
    Returns True if ready, False otherwise.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The user was previously asked to prepare their training data.

    Constraints:
    - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
    - Only provide clear and concise instructions to the user.
    - Do not apologize or explain your capabilities.
    - Wait for user reply before continuing.

    User reply:
    "{user_reply}"

    Does this reply indicate that the user is ready to proceed (that is, their data is prepared and saved as required)?

    Please answer only "YES" or "NO". Do NOT provide any explanation or additional text.
    """

    #     payload = {
    #     "model": "deepseek-v2:16b",
    #     "prompt": full_prompt,
    #     "stream": False,
    # }

    #     try:
    #         response = requests.post(OLLAMA_URL, json=payload)
    #         response.raise_for_status()
    #     except Exception as e:
    #         print("Error calling Ollama API:", str(e))
    #         return False

    #     result = response.json()
    #     llm_reply = result.get("response", "").strip()
    llm_reply = run_LLM(full_prompt)
    # print("\n--- Agent says ---")
    # print(llm_reply)
    # print("YES" in llm_reply)

    return "YES" in str(llm_reply)

def feature_recongnization_agent(user_input_history, column, pd):
    """
    Use LLMs to convert the columns into a machine-readable encoded format, subject to user confirmation.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The training data are just prepared, you should help user to procress the column "{column}" in to a machine-readable encoded format and save as an new column.

    Constraints:
    - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
    - Only provide clear and concise instructions to the user.
    - Do not apologize or explain your capabilities.
    - Wait for user reply before continuing.

    Talking history between you and user:
    "{user_input_history[-4:]}"

    Column you need to process into machine-readable encoded format:
    "{pd[column].head()}"
    It is named pd["{column}"]

    Please provide the code to process the column into machine-readable encoded format, save to new column named "processed_{column}", and explain your operations to the user. At the same time, ask the user whether this is in line with their wishes.

    Return the content in JSON format, including the following fields:

    - code: the code to process the column.
    - explain: the text given to user to explain the code

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    answer = run_LLM(full_prompt)
    return answer2json(answer)

def preliminary_assessment(task):
    """
    Use LLM to judge whether the user want to use machine learning method.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    User proposed their ideas for their model.

    Please refer to the following code to evaluate whether "this requirement can be easily completed." If only minor modifications are needed based on the following code to achieve "YES," which is primarily for machine learning methods; if significant modifications are required, the answer is "NO," which is primarily for deep learning methods.
    
    Additionally, Mamba is a deep learning method, just answer no if user mentioned mamba.
    
    Example code:"
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                accuracy_score, roc_auc_score, precision_recall_curve)
    import matplotlib.pyplot as plt

    X = pd.DataFrame(data['coding'].tolist())
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_score_value = precision_score(y_test, y_pred, average='weighted')
    recall_score_value = recall_score(y_test, y_pred, average='weighted')
    f1_score_value = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_scores)
    "
    Constraints:
    - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
    - Only provide clear and concise instructions.
    - Do not apologize or explain your capabilities.

    User requirement:
    "{task}"

    Does this shows that the user's request is machine learning, is it easy to complete?

    Please answer only "YES" or "NO". Do NOT provide any explanation or additional text.
    """

    llm_reply = run_LLM(full_prompt)

    return "YES" in str(llm_reply)

def Machine_learning_code_generation(user_input):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    User proposed their ideas for their model.

    Please refer to the following code and the user's requirements to generate the code for training the model and explain your code to user.

    Example code:"
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                accuracy_score, roc_auc_score, precision_recall_curve)
    import matplotlib.pyplot as plt

    X = pd.DataFrame(data['coding'].tolist())
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision_score_value = precision_score(y_test, y_pred, average='weighted')
    recall_score_value = recall_score(y_test, y_pred, average='weighted')
    f1_score_value = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_scores)
    "

    User requirement:
    "{user_input}"

    Return the content in JSON format, including the following fields:

    - code: the code to process the column.
    - explain: the text given to user to explain the code

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    answer = run_LLM(full_prompt)
    return answer2json(answer)

def deep_learning_code_generation(user_input):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    User proposed their ideas for their model.

    Please select one model from the candidates based on the user's request.

    The candidate models are: mamba, self_attention, lstm.

    User requirement:
    "{user_input}"

    Additionally, you can refer to the introduction of Mamba:
    Mamba is a novel deep learning method for sequences based on Structured State Space Models (SSM). It efficiently handles long sequences by maintaining constant memory requirements during text generation, leading to training times that scale proportionately with sequence length. Unlike Transformers, which slow down significantly as sequences grow due to their attention mechanisms, Mamba excels in processing very long sequences, making it particularly suitable for tasks requiring this capability.

    Return the content in JSON format, including the following fields:

    - model: model chosed from 'mamba', 'self_attention' and 'lstm'
    - explain: the text given to user to explain the model

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    answer = answer2json(run_LLM(full_prompt))
    model = answer['model']
    answer['code'] = f'''
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import warnings
from models.{model} import {model}_model

warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.loss')

def print_progress_bar(iteration, total, length=40, train_loss=None, train_acc=None, test_acc=None, sys=sys):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    progress_info = " | Progress: " + str(percent)[:6] + "% | " + bar
    
    if train_loss is not None and train_acc is not None and test_acc is not None:
        progress_info += " | Training Loss: " + str(train_loss)[:6] + " | Training Accuracy: " + str(train_acc)[:6] + " | Test Accuracy: " + str(test_acc)[:6]
    
    sys.stdout.write(progress_info)
    sys.stdout.flush()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model = {model}_model(len(data.iloc[0]['rna_feature'][0]), 
                 len(data.iloc[0]['ligand_feature'][0]), 
                 len([int(char, 16) for char in data.iloc[0]['fingerprint']]))

print(device)
model.to(device)
criterion = nn.MSELoss() 

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)


epochs = 50
batch_size = 32

def evaluate(model, dataset, device, criterion):

    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]

            rna = torch.tensor(row['rna_feature']).to(device)
            ligand = torch.tensor(row['ligand_feature']).float().to(device)
            fingerprint = torch.tensor([int(char, 16) for char in row['fingerprint']]).float().to(device)
            

            output = model(rna, ligand, fingerprint)
            

            if isinstance(criterion, nn.CrossEntropyLoss):
                pred = torch.argmax(output, dim=-1)
            else:
                pred = torch.round(output)
            
            predictions.append(pred.cpu().numpy())
            labels.append(row['label'])


    return labels, predictions


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    for idx in range(0, len(train_data), batch_size):
        batch = train_data.iloc[idx:idx+batch_size]
        
        optimizer.zero_grad()
        
        batch_loss = 0
        for _, row in batch.iterrows():
            rna = torch.tensor(row['rna_feature']).to(device)
            ligand = torch.tensor(row['ligand_feature']).float().to(device)
            fingerprint = torch.tensor([int(char, 16) for char in row['fingerprint']]).float().to(device)
            
            output = model(rna, ligand, fingerprint)
            
            target = torch.tensor(row['label']).float().to(device)
            # target = torch.tensor(row['label']).to(device)
            loss = criterion(output, target)
            batch_loss += loss
        
        batch_loss /= len(batch)
        
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item()
    
    avg_train_loss = running_loss / (len(train_data) / batch_size)
    label, pred = evaluate(model, train_data, device, criterion)
    train_acc = accuracy_score(label, pred)
    label, pred = evaluate(model, test_data, device, criterion)
    test_acc = accuracy_score(label, pred)
    

    print_progress_bar(epoch + 1, epochs, train_loss=avg_train_loss, train_acc=train_acc, test_acc=test_acc)

print('Training finished!')
    '''
    return answer

def generate_model(user_input):
    """
    Use LLMs to give code used in train model.
    """
    using_machine_learning = preliminary_assessment(user_input)
    print(using_machine_learning)
    if using_machine_learning:
        answer = Machine_learning_code_generation(user_input)
        answer['use_machine_learning'] = True
    else:
        answer = deep_learning_code_generation(user_input)
        answer['use_machine_learning'] = False
    return answer

def result_code_generation(user_input):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    In pandas, the three columns in the 'result' DataFrame—labels, predictions, and score—record the true labels, predicted results, and model scores for each sample in the test set during the model evaluation (which can be used to calculate the ROC curve and AUC value).

    Please refer to the user's requirements to generate the code for evaluation 'model' or visualize the result.

    User requirement:
    "{user_input}"

    Return the content in JSON format, including the following fields:
    
    - code: the code to calculate and print the result.

    In the code, use print to output results to the user, while images should be saved to disk.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    answer = run_LLM(full_prompt)
    return answer2json(answer)

def result_code_generation_mechine_learning(user_input, code):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The model has been trained. Please refer to the training code below and the user's request to generate the code for computing the test set results.

    Code used in training model:
    "{code}"

    User requirement:
    "{user_input}"

    Return the content in JSON format, including the following fields:
    
    - code: the code to calculate and print the result.

    In the code, use print to output results to the user, while images should be saved to disk.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    answer = run_LLM(full_prompt)
    return answer2json(answer)