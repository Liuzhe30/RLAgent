# RLAgent
An Interactive Agent Framework for Adaptive Modeling of RNA Region-Ligand Interactions

<p align="center"><img width="100%" src="images/pipeline.png" /></p>

## Installation

1. Install Python environment:
```bash
conda create -n rlagent python=3.9
conda activate rlagent
git clone https://github.com/Liuzhe30/RLAgent
cd RLAgent
```
2. Install required Python packages:
```bash
pip install -r requirements.txt
pip install e .
```
3. Install Ollama:  
Follow instructions at https://ollama.com/download  
For Linux:
```bash
curl https://ollama.com/install.sh | sh
```
4. Pull LLM:
```bash
ollama pull deepseek-v2:16b
```

## Usage
Run the agent demo:
```bash
python scripts/run_agent_demo.py
```

## RLAgent Case Demonstrations

Below are two representative case workflows demonstrating how RLAgent performs region-level RNA–ligand interaction modeling through fully natural language–driven execution, from dataset preparation to model evaluation.

---

### Case 1: Dialogue-driven Initialization and Feature Processing
![Case 1](images/case1.png)  
*RLAgent guides the user through dataset preparation, confirms the modeling plan, and processes RNA sequences, region masks, and ligand information. Features are generated using optional RNA foundation model embeddings, optional RAG-based knowledge integration, one-hot encoding, and molecular fingerprints.*

---

### Case 2: Model Construction and Evaluation
![Case 2](images/case2.png)  
*Processed features are combined and transformed using Mamba modules, pooled, and merged via a multi-linear layer to produce predictions. The workflow includes real-time reporting of training metrics, precision calculation on the test set, and ROC–AUC performance visualization.*