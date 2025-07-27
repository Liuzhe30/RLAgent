# rlagent/agent_controller.py
import torch
import torch.nn as nn
from rlagent.llm_tools import ask_llm_stage, judge_user_ready, deeplearning_build, generate_model, result_code_generation, result_code_generation_mechine_learning, absolute_exec
from rlagent.data_utils import check_and_process_data
import pandas as pd

def predict(model, sample, features):
    input = []
    for feature in features:
        input.append(torch.from_numpy(sample[feature]))
    return model(*input)
def evaluate_result(model, dataset, features):

    criterion = nn.MSELoss()
    model.eval()
    scores = []
    predictions = []
    labels = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            
            # 获取预测结果
            output = predict(model, row, features)
            scores.append(output.cpu().item())
            # 根据任务类型处理输出
            if isinstance(criterion, nn.CrossEntropyLoss):
                pred = torch.argmax(output, dim=-1)
            else:  # 回归任务可以四舍五入
                pred = torch.round(output)
            
            predictions.append(pred.cpu().item())
            labels.append(row['label'])
    # print(predictions)
    # 计算准确率
    # return accuracy_score(labels, predictions)
    return labels, predictions, scores


class AgentController:

    def __init__(self):
        print("Initializing RLAgent Controller...")

    def run_pipeline(self):
        user_reply = ""

        # Stage 1 - Ask user to prepare data
        msg = ask_llm_stage("Explain to the user how to prepare their training dataset. Ask user to prepare input CSV in datasets/train.csv with correct format", user_reply)

        # Wait for user confirmation (loop, judged by LLM)
        ready = False
        while not ready:
            user_input = input("\n[Your reply] → ").strip()
            ready = judge_user_ready(user_input)

            if ready:
                print("\nUnderstood. Starting to process your training data, please wait...")
            else:
                print("Waiting for your confirmation. Please type when your data is ready...")



        # # Stage 2.0 - order elemant and label used


        # Stage 2.3 - Check data

        # Improve later
        # columns_used = ["ligand", "rna_sequence"]
        # using_RAG = True

        data, feature, label = check_and_process_data("./datasets/dataset_annotation.csv", "./datasets/demo_processed.csv")
        if data is None:
            print("Data processing failed. Please check your CSV file and try again.")
            return
        else:
            print("Data processing completed. Saved as ./datasets/demo_processed.csv.")
        
        # Stage 3 - Model build and exec
        model_code = deeplearning_build(data, feature, label)
        houxu = f'''

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
model = sum_model()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
epochs = 50
batch_size = 32
train_data, test_data = train_test_split(data, test_size=0.2)
def predict(model, sample):
    return model({', '.join(["torch.from_numpy(sample['" + i + "']).to(device)" for i in feature])})

    
def evaluate(model, dataset):
    """评估模型在给定数据集上的准确率"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]

            output = predict(model, row)
            
            pred = torch.round(output)
            
            predictions.append(pred.cpu().numpy())
            labels.append(row['{label}'])

    # 计算准确率
    return accuracy_score(labels, predictions)

# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # 随机打乱训练数据
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    for idx in range(0, len(train_data), batch_size):
        batch = train_data.iloc[idx:idx+batch_size]
        
        # 清零梯度
        optimizer.zero_grad()
        
        batch_loss = 0
        for _, row in batch.iterrows():

            output = predict(model, row)
            
            # 计算损失
            target = torch.tensor(row['{label}']).float().to(device)
            loss = (output - target) ** 2
            batch_loss += loss
        
        # 平均损失
        batch_loss /= len(batch)
        
        # 反向传播和优化
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item()
    
    # 计算平均训练损失
    avg_train_loss = running_loss / (len(train_data) / batch_size)
    
    # 评估模型
    train_acc = evaluate(model, train_data)
    test_acc = evaluate(model, test_data)
    
    print(f'Epoch {{epoch+1}}/{{epochs}}')
    print(f'Training Loss: {{avg_train_loss:.4f}} | Training Accuracy: {{train_acc:.4f}} | Test Accuracy: {{test_acc:.4f}}')
    print('-' * 50)

print('Training finished!')
'''
        namespace = {}
        namespace['data'] = data
        exec(model_code + houxu, namespace)
        model = namespace['model']
        test_data = namespace['test_data']
        labels, predictions, score = evaluate_result(model, test_data, feature)
        ddd = {
            'labels': labels,
            'predictions': predictions,
            'score': score
        }
        result = pd.DataFrame(ddd)
        while True:
            user_input = input("\n[Your request, 'exit' to end this stage] → ").strip()
            if user_input == 'exit':
                break
            result = result_code_generation(user_input)
            absolute_exec(result['code'], user_input)
        # Stage 3 - Model selection and run model
        # user_input = input("\n[Discrib the model you want] → ").strip()
        # output = generate_model(user_input)
        # print(output['explain'])
        # namespace = {}
        # namespace['data'] = data
        # exec(output['code'], namespace)
        # # Stage 4 - Result reading and visualization
        # if output['use_machine_learning']:
        #     while True:
        #         user_input = input("\n[Your request, 'exit' to end this stage] → ").strip()
        #         if user_input == 'exit':
        #             break
        #         result = result_code_generation_mechine_learning(user_input, output['code'])
        #         # exec(result['code'])
        #         absolute_exec(result['code'], user_input)
        # else:
        #     # print("Global variables:", list(globals().keys()))
        #     # print("Local variables:", list(locals().keys()))
        #     # exec('model = nn.Linear(10, 5)', locals())
        #     # exec('nann = nn.Linear(10, 5)', locals())
        #     # print("Global variables:", list(globals().keys()))
        #     # print("Local variables:", list(locals().keys()))
        #     # print(next(model.parameters()).shape)
        #     print("Local variable names:", list(locals().keys()))
        #     model = namespace['model']
        #     test_data = namespace['test_data']
        #     labels, predictions, score = evaluate_result(model, test_data)
        #     ddd = {
        #         'labels': labels,
        #         'predictions': predictions,
        #         'score': score
        #     }
        #     result = pd.DataFrame(ddd)
        #     while True:
        #         user_input = input("\n[Your request, 'exit' to end this stage] → ").strip()
        #         if user_input == 'exit':
        #             break
        #         result = result_code_generation(user_input)
        #         absolute_exec(result['code'], user_input)
        print("\nRLAgent pipeline completed!")
