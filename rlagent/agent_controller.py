# rlagent/agent_controller.py
import torch
import torch.nn as nn
from rlagent.llm_tools import ask_llm_stage, judge_user_ready, generate_model, result_code_generation
from rlagent.data_utils import check_and_process_data
import pandas as pd
def evaluate_result(model, dataset):
    """评估模型在给定数据集上的准确率"""
    criterion = nn.MSELoss()
    model.eval()
    scores = []
    predictions = []
    labels = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            # 准备输入数据
            rna = torch.tensor(row['rna_feature']).to(device)
            ligand = torch.tensor(row['ligand_feature']).float().to(device)
            fingerprint = torch.tensor([int(char, 16) for char in row['fingerprint']]).float().to(device)
            
            # 获取预测结果
            output = model(rna, ligand, fingerprint)
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
        # # Exiting elemant select
        # ready = False
        # user_messages = []
        # while not ready:
        #     user_input = input("\n[Your reply] → ").strip()
        #     ready = judge_user_ready(user_input)

        #     if ready:
        #         print("\nUnderstood. Starting to process your training data, please wait...")
        #     else:
        #         print("Waiting for your confirmation. Please type when your data is ready...")


        # Stage 2.3 - Check data

        # Improve later
        columns_used = ["ligand", "rna_sequence"]
        using_RAG = True

        data = check_and_process_data("./datasets/train.csv", "./datasets/demo_processed.csv", columns_used, using_RAG)
        if data is None:
            print("Data processing failed. Please check your CSV file and try again.")
            return
        else:
            print("Data processing completed. Saved as ./datasets/demo_processed.csv.")
        

        # Stage 3 - Model selection and run model
        user_input = input("\n[Discrib the model you want] → ").strip()
        output = generate_model(user_input)
        print(output['explain'])
        device = 'cuda:2'
        exec('print(device)\n' + output['code'])
        # Stage 4 - Result reading and visualization
        if output['use_machine_learning']:
            print('done')
        else:
            labels, predictions, score = evaluate_result(model, test_data)
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
                exec(result['code'])
        print("\nRLAgent pipeline completed!")