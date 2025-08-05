# rlagent/llm_tools.py

import httpx
from typing import Optional
import asyncio
import json
import numpy as np
import torch

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

def run_LLM_json_auto_retry(full_prompt):
    answer = run_LLM(full_prompt)
    try:
        result_json = answer2json(answer)
    except Exception as e:
        result_json = run_LLM_json_auto_retry(full_prompt)
    return result_json

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


epochs = 2
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
    return run_LLM_json_auto_retry(full_prompt)

def result_code_generation_mechine_learning(user_input):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The model has been trained. Please refer to the training code below and the user's request to generate the code for computing the test set results.

    Code used in training model:
    "
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
    
    - code: the code to calculate and print the result.

    In the code, use print to output results to the user, while images should be saved to disk.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    return run_LLM_json_auto_retry(full_prompt)

def query_check(code, query):
    """
    verify that the code along with user's query
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The code has been generated according to the user's request. Please evaluate whether it meets the user's requirements. If it does not, please make the necessary corrections.
    Unless explicitly stated by the user, the code should not generate sample data. If the code includes sample data, it does not meet the user's request and needs to be corrected.

    Code:
    "{code}"

    User's query or code description:
    "{query}"

    Additionally：
    The num_head for self_attention attention can be set to 1.
    
    Return the content in JSON format, including the following fields:
    - meet: A boolean variable to indicate whether the code meets the user's request.
    - code: The corrected code when it does not meet the user's request, optional.

    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    return run_LLM_json_auto_retry(full_prompt)

def debug_agent(code, query, error, variation = None):
    if variation is not None:
        variation = {q: k.shape for q,k in variation}
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The code has been generated according to the user's request or describe. However, an error occurred during execution. Please correct the code.
    
    Code:
    "{code}"

    User's query or code description:
    "{query}"

    Error or Dissatisfied needs:
    "{error}"

    Current variable and their shape (which may sometimes be provided)：
    "{variation}"
    The following content may be related to undefined functions：
    "
from models.mamba import mambalayer
embedding_dim = 50
device = 'cuda:2'
input = torch.randn(100, embedding_dim).to(device)
layer = mambalayer(embeddingdim = embedding_dim).to(device)
output = layer(input)

from models.self_attention import bertlayer
embedding_dim = 50
input = torch.randn(100, embedding_dim)
layer = bertlayer(embeddingdim = embedding_dim, num_head = 5) 
output = layer(input)
    "
    Try to avoid naming new variables in most case and use the same variable name to demonstrate the flow of data.
    Since all data has been provided, please strictly ensure that the code does not generate sample data or use variables for demonstration, such as "feature_1 = torch.randn(42, embedding_dim)"
    Return the content in JSON format, including the following fields:
    - code: The corrected code when it does not meet the user's request, optional.

    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text. Do NOT generate example data for 'result'. 
    """
    result = run_LLM_json_auto_retry(full_prompt)
    return result['code']


def absolute_exec(code, query):
    """
    Execute the code in a loop and fix any errors.
    """
    
    result = query_check(code, query)
    if result['meet']:
        try:
            exec(code) #check but not exec should be better
        except Exception as e:
    
            error_message = f"Error: {str(e)}\n"
            error_message += f"Line number: {e.__traceback__.tb_lineno}"
            code = debug_agent(code, query, error_message)
            absolute_exec(code, query)

    else:
        absolute_exec(result['code'], query)
        
    return None


def column_process_json_generation(full_prompt):
    result = run_LLM_json_auto_retry(full_prompt)
    if 'description' in result.keys() and 'code' in result.keys() :
        return result
    else:
        return column_process_json_generation(full_prompt)
    
def init_column_process(column, new_column_name):
    # add static fonction to it
    init_full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The user has prepared the data. Please provide suggestions for processing the current column into a feature matrix or feature vector

    The data is stored in a Pandas DataFrame named 'data'.

    Do NOT generate any sample data in code
    
    Column need to be processed:
    "{str(column.head())}"
    
    Processed feature should be converted to NumPy saved as a new column named:"{new_column_name}", each row should be a NumPy array.

    You can refer to the preset methods provided below. The functions for these methods are already defined, so do NOT generate example functions when using them.
    Code for preset methods:
"
from rlagent.preset_method import process_RNA

data['new_column'] = data['rna_sequence'].apply(process_RNA) # Use the foundation model: RNA-FM to extract features from RNA, extracting embeddings of dimension 640 for each nucleotide.

from rlagent.preset_method import ligand2onehot, get_dict
words = get_dict(data['smile'].tolist())
data['ligand_feature'] = data['smile'].apply(lambda x: ligand2onehot(x, words)) # "Perform one-hot encoding on the SMILES string. "

from rlagent.preset_method import get_fingerprint
data['fingerprint'] = data['smile'].apply(get_fingerprint) # Use the interface of pubchem.ncbi.nlm.nih.gov to query the fingerprint of the SMILES.
"
    Don't forget the 'import' code
    Return the content in JSON format, including the following fields:
    
    - description: description of the code and the processing logic.
    - code: Code to process 

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    return column_process_json_generation(init_full_prompt)

def column_process_adjust(column, code, query, new_column_name):
    init_full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The processing solution has been provided to the user, but adjustments are needed due to user dissatisfaction or runtime errors.
        
    Column need to be processed:
    "{str(column.head())}"
    
    Processed feature should saved as a new column named:"{new_column_name}"

    Code need to be processed:
    "{code}"

    User's query:
    "{query}"
    
    Return the content in JSON format, including the following fields:
    
    - description: description of the code and the processing logic.
    - code: Code to process 

    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    return column_process_json_generation(init_full_prompt)


def process_absolute_exec(code, query, column_name, data):
    """
    Execute the code in a loop and fix any errors.
    """
    
    result = query_check(code, query)
    if result['meet']:
        try:
            namespace = {}
            namespace['data'] = data
            exec(code, namespace) #check but not exec should be better
            data = namespace['data']
            if not column_name in data.columns:
                data = process_absolute_exec(code, f'processed feature are not saved in column named: "{column_name}"', column_name, data)
            if not data[column_name].apply(lambda x: isinstance(x, np.ndarray)).all():
                data = process_absolute_exec(code, f'processed feature are not saved as numpy', column_name, data)
        except Exception as e:
    
            error_message = f"Error: {str(e)}\n"
            error_message += f"Line number: {e.__traceback__.tb_lineno}"
            print(error_message)
            code = debug_agent(code, query, error_message)
            data = process_absolute_exec(code, query, column_name, data)

    else:
        print('code adjust')
        print(result['code'], query)
        data = process_absolute_exec(result['code'], query, column_name, data)
        
    return data

def judge_user_satisfied(user_reply, asking) -> bool:
    """
    Use LLM to judge whether the user's reply indicates that they are ready.
    Returns True if ready, False otherwise.
    """
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    You have provided a solution for the user. Please assess the user's response to determine if they are satisfied about the solution and don't need adjust anymore.

    Constraints:
    - Do NOT output "Think:", "Thought:", "<think>", "Reasoning", or any meta-comments.
    - Only provide clear and concise instructions to the user.
    - Do not apologize or explain your capabilities.
    - Wait for user reply before continuing.

    User reply about "{asking}":
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

def column_process(column, data, new_column_name):
    plan = init_column_process(data[column], new_column_name)

    while True:
        description = plan['description']
        code = plan['code']
        print(description)
        asking = "\n[Is it reasonable?] → "
        responce = input(asking).strip()
        if judge_user_satisfied(responce, asking):
            process_absolute_exec(code, description, new_column_name, data)
            
            break
        else:
            print('plan adjusting')
            plan = column_process_adjust(data[column], code, description, new_column_name)
               
    return data

def feature_recognition(data, full_prompt):
    result = run_LLM_json_auto_retry(full_prompt)
    # print(result)
    try:
        text = result['explain']
        if all(col in data.columns for col in result['feature']) & (result['label'] in data.columns):
            return result
        else:
            print('unrecongnize_column', result['feature'], result['label'])
            return feature_recognition(data, full_prompt)
    except Exception as e:
        print(e)
        return feature_recognition(data, full_prompt)

def Feature_Recognition(data):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The user has prepared the data. Please select some columns that are suitable as features and which column is suitable as the label based on the user's prepared data.
    User's data:
    "{str(data.sample(n=5))}"

    Return the content in JSON format, including the following fields:
    
    - feature: Record the column names selected to be used as features in a list.
    - label: Column name suitable to be used as label
    - explain: Explain to user about the reasion of your decision 

    In most cases, 2 to 4 of the most suitable features are sufficient.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    return feature_recognition(data, full_prompt)

def Feature_adaption(data, plan, description):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The data and feature identification plan is ready. Please adjust the  about feature and label based on the user's description.
    User's data:
    "{str(data.sample(n=5))}"
    
    Tentative plan:
    "{plan}"

    User's description:
    "{description}"
    
    Return the content in JSON format, including the following fields:
    
    - feature: Record the column names selected to be used as features in a list.
    - label: Column name suitable to be used as label
    - explain: Explain to user about the reasion of your decision 

    In most cases, 2 to 4 of the most suitable features are sufficient.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.
    """
    return feature_recognition(data, full_prompt)

def Laywer_generation(description, Variable):
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    Please build a partial neural network based on the user's description. and introduce it to the user.
    
    User's describe:
    "{description}"

    Current variable:
    "{Variable}"

    Refer to the following self-attention code.
"
from models.self_attention import bertlayer
embedding_dim = 50
input = torch.randn(100, embedding_dim)
layer = bertlayer(embedding_dim, embedding_dim*4, num_head = 5)
layer(input).shape
"
    Refer to the following Mamba code. Mamba performs the same tasks as self-attention, with comparable performance but higher inference speed and lower memory usage.
"
from models.mamba import mambalayer
embedding_dim = 50
device = 'cuda:2'
input = torch.randn(100, embedding_dim).to(device)
layer = mambalayer(embedding_dim, embedding_dim*4).to(device)
layer(input).shape
"    
    Return the content in JSON format, including the following fields:
    
    - code: Record the column names selected to be used as features in a list.
    - text: Explaintion that make introduce to user.

    All variables have been defined; please do NOT generate sample data..
    Please respond with JSON only. Do NOT provide any other explanation or additional text.

    """
def check_exec(code, query, variation):
    """
    Execute the code in a loop and fix any errors.
    """
    
    result = query_check(code, query)
    if result['meet']:
        try:
            exec(code, variation) #check but not exec should be better
            variation = {name: obj for name, obj in variation.items() if hasattr(obj, 'shape')}
        except Exception as e:
            print('err________________________')
            error_message = f"Error: {str(e)}\n"
            error_message += f"Line number: {e.__traceback__.tb_lineno}"
            print(code)
            print(error_message)
            code = debug_agent(code, query, error_message)
            code, variation = check_exec(code, query, variation)

    else:
        print('not_meet________________________________')
        print(code)
        print(query)
        code, variation = check_exec(result['code'], query, variation)
        
    return code, variation

def laywer_generation_once(description, variation):
    variable = {}
    for single_variation in variation:
        variable[single_variation] = variation[single_variation].shape
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    Please build a partial neural network or sever step based on the user's description. and introduce it to the user.
    
    User's describe:
    "{description}"

    Current variable and their shape:
    "{variable}"

    Refer to the following self-attention code.
"
from models.self_attention import bertlayer
embedding_dim = 50
input = torch.randn(100, embedding_dim)
layer = bertlayer(embedding_dim, num_head = 5) 
output = layer(input)
"
    Refer to the following Mamba code. Mamba performs the same tasks as self-attention, with comparable performance but higher inference speed and lower memory usage.
    Do NOT use mamba if user mentioned self_attention
"
from models.mamba import mambalayer
embedding_dim = 50
device = 'cuda:2'
input = torch.randn(100, embedding_dim).to(device)
layer = mambalayer(embedding_dim).to(device)
output = layer(input)
"    
    Try to avoid naming new variables in most case and use the same variable name to demonstrate the flow of data.
    Return the content in JSON format, including the following fields:
    
    - code: Record the column names selected to be used as features in a list.
    - text: Explaintion that make introduce to user.

    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.

    """
    result = run_LLM_json_auto_retry(full_prompt)
    # result['code'], result['variation'] = check_exec(result['code'], result['text'], variation)
    result['code'], result['variation'] = check_exec(result['code'], description, variation)
    return result

def return_select(label, variation, discribtion):
    variable = {}
    for single_variation in variation:
        variable[single_variation] = variation[single_variation].shape
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.

    The internal logic of the deep learning model has now been finalized. 
    Please select the output based on the user's appeal and determine if the model's output can be appropriately reshaped into the form of labels. 
    If so, please also provide the code for the reshaping process.
    
    User's describe (if any, ignore and select the output yourself):
    "{discribtion}"

    Current variable and their shape:
    "{variable}"

    Labels and it's shapes that need to be fitted (An empty shape represents a single value.):
    "{label}"
    
    Distinguish between the variable names for the result and the label.
    
    Return the content in JSON format, including the following fields:
    - Reasonable：A boolean variable indicating whether the task is reasonable should be set to False if the output is difficult to process into the shape of the label or if the user's description is vague.
    - output: appropriate output selected from the provided variables.
    - code: code that reshape the output into the form of labels.
    - text: Explaintion that make introduce to user.

    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.

    """
    result = run_LLM_json_auto_retry(full_prompt)
    result['code'], _ = check_exec(result['code'], result['text'], variation)
    return result

def deeplearning_build(data, features, label):
    va = {}
    codes = []
    sample = data.iloc[0]
    need_adjust = False
    for feature in features:
        va[feature] = torch.from_numpy(sample[feature])
    while True:
        #Layer recommendation、
        asking = "\n[discrib the Operation you want made] → "
        responce = input(asking).strip()
        if judge_user_satisfied(responce, asking):
            #chose output
            while True:
                asking = "\n[Select a variable as model output please] → "
                responce = input(asking).strip()
                if responce == 'adjust':
                    need_adjust = True
                    break
                result = return_select({label: data.iloc[0][label]}, va, responce)
                asking = "\n[Is that reasonable?] → "
                responce = input(asking).strip()
                if judge_user_satisfied(responce, asking):
                    break
                else:
                    print('Your description is not clear enough, or the return value is difficult to process into the same shape as the label. Please describe it again.')
                    print("here are agent's discribtion:")
                    print(result['text'])
                    print("input 'adjust' back to model building")
            if need_adjust:
                need_adjust = False
                continue
            break
        else:
            va = {k: v for k, v in va.items() if isinstance(v, torch.Tensor) and v.shape}
            result = laywer_generation_once(responce, va)
            print(result['text'])
            asking = "\n[is that resonable?] → "
            responce = input(asking).strip()
            if judge_user_satisfied(responce, asking):
                codes.append(result['code'])
                va = result['variation']
                print('Operation adoption')
            else:
                print('Operation rejected')
    sum_model = f'''
import torch
import torch.nn as nn

# Define the neural network
class sum_model(nn.Module):
    def __init__(self):
        super(sum_model, self).__init__()
        print('model building')

    def forward(self, {', '.join(features)}):
        return None

'''
    for code in codes:
        va = {}
        for feature in features:
            va[feature] = torch.from_numpy(sample[feature])
        result = assembly_model(sum_model, code, va)
        sum_model = assembly_model(sum_model, result['code'], va)
    return sum_model

def check_exec_assembly(code, query, variation, features):
    """
    Execute the code in a loop and fix any errors.
    """
    
    result = query_check(code, query)
    if result['meet']:
        try:
            exec(code + f"\nmodel = sum_model() \nmodel({', '.join(features)})", variation) #check but not exec should be better
            variation = {name: obj for name, obj in variation.items() if hasattr(obj, 'shape')}
        except Exception as e:
            print('err________________________')
            error_message = f"Error: {str(e)}\n"
            error_message += f"Line number: {e.__traceback__.tb_lineno}"
            print(code)
            print(error_message)
            code = debug_agent(code, query, error_message, variation = variation)
            code, variation = check_exec_assembly(code, query, variation, features)

    else:
        print('not_meet________________________________')
        print(code)
        print(query)
        code, variation = check_exec_assembly(result['code'], query, variation, features)
    return code, variation

def assembly_model(current, addition, variation, features):
    task = f'''
    Please integrate the code snippets, which include several layers and their data flow, into a complete deep learning model.
    Pay close attention to the __init__ and forward sections. Define the layers, especially those that need tuning, as self attributes in the __init__ part, while placing the remaining components in the forward method.

    The current model (which may be empty).
    "{current}"
    
    Layers and operations to be added:
    "{addition}"
    '''


    
    full_prompt = f"""
    You are RLAgent, an assistant helping the user with RNA-Ligand modeling.
    
    {task}
    
    Don't worry about the return for now
    Return the content in JSON format, including the following fields:
    -code

    All variables have been defined; please do NOT generate sample data.
    Please respond with JSON only. Do NOT provide any other explanation or additional text.

    """
    result = run_LLM_json_auto_retry(full_prompt)
    result['code'], _ = check_exec_assembly(result['code'], task, variation, features)
    return result
