# rlagent/llm_tools.py

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

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

    payload = {
    "model": "deepseek-v2:16b",
    "prompt": full_prompt,
    "stream": False,
}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    except Exception as e:
        print("Error calling Ollama API:", str(e))
        return "[Error: failed to contact LLM]"

    result = response.json()
    llm_message = result.get("response", "").strip()

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

    payload = {
    "model": "deepseek-v2:16b",
    "prompt": full_prompt,
    "stream": False,
}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    except Exception as e:
        print("Error calling Ollama API:", str(e))
        return False

    result = response.json()
    llm_reply = result.get("response", "").strip()

    print("\n--- Agent says ---")
    print(llm_reply)

    return "yes" in llm_reply