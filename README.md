# LLM Survey Response Time Prediction

This repository contains a simulation script for predicting **median response times** of survey questions using a Large Language Model (LLM).  
The goal of the project is to compare different **prompt engineering strategies** and assess their impact on predicted response times.

---

## Project Structure


Project Structure

run_simulation.py
data/questions.json
prompts/baseline.txt
prompts/example.txt
prompts/chain_of_thought.txt
results/


---

## File Descriptions

### `run_simulation.py`

Main execution script.  
Loads survey questions and prompt templates, calls the LLM, parses model outputs, and stores predicted response times.

---

### `data/questions.json`

JSON file containing survey questions.  
Each entry includes:
- a question ID
- the full question text
- a coded domain describing the response options

---

### `prompts/`

Contains prompt templates used for different prompt engineering strategies:

- **`baseline.txt`**  
  Direct estimation of response time without examples.

- **`example.txt`**  
  Estimation guided by example questions with known response times.

- **`chain_of_thought.txt`**  
  Step-by-step decomposition into reading, comprehension, and selection time.

---

### `results/`

Output directory.  
The script writes one or more CSV files containing predicted response times for each survey question and prompt technique.


---

## LLM Backend and Extensibility

The simulation script is intentionally designed to keep LLM interaction **simple and transparent**.  
All model-specific logic is contained in a single function:

```python
call_llm(prompt)
