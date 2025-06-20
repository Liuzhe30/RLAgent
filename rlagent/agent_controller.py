# rlagent/agent_controller.py

from rlagent.llm_tools import ask_llm_stage, judge_user_ready
from rlagent.data_utils import check_and_process_data

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

        # Stage 2 - Check and process data
        data = check_and_process_data("./datasets/train.csv", "./datasets/demo_processed.csv")
        if data is None:
            print("Data processing failed. Please check your CSV file and try again.")
            return
        else:
            print("Data processing completed. Saved as ./datasets/demo_processed.csv.")

        print("\nRLAgent pipeline completed!")

