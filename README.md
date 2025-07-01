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