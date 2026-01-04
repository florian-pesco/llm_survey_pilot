import pandas as pd
import json
import re

def call_llm(prompt):
    """
    Plug in LLM call.
    """
    
    return ollama(prompt)

def ollama(prompt):
    import requests
    
    MODEL = "llama3.2:3b"
    TEMPERATURE = 0.7
    
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "stream": False
        }
    )

    raw = r.json()["response"].strip()
    print(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Could not parse JSON:", raw)
        return None
    time = data.get("total_predicted_duration", -1)
    print(time)
    return time

def load_prompt(filename):
    with open(f'prompts/{filename}', 'r') as f:
        return f.read()
    
def get_question_ids(questions_json):
    ids = []
    for q in questions_json:
        ids.append(q['id'])
    return ids

def get_question_data(q_id, questions_json):
    for q in questions_json:
        if q['id'] == q_id:
            return q
    return None

def format_domain(domain_dict):
    return ", ".join([f"{k}: {v}" for k, v in domain_dict.items()])

NUM_ITER = 1

response_format = f"""
Respond ONLY in valid JSON:
{{
  "explanation": "string",
  "total_predicted_duration": number
}}
"""


def main():
    # Load questions with code domains
    with open('data/questions.json', 'r') as f:
        questions = json.load(f)

    # Load prompt templates
    template_baseline = load_prompt('baseline.txt')
    template_example = load_prompt('example.txt')
    template_chain_of_thought = load_prompt('chain_of_thought.txt')

    results = []

    target_questions = get_question_ids(questions)
    for q_id in target_questions:
        q_info = get_question_data(q_id, questions)
        q_text = q_info['question_text']
        q_dom = format_domain(q_info['code_domain']) 
        
        print("question: "+ q_id + ": "+ q_text)

        for i in range(NUM_ITER):
            print("Baseline: ")
            # Prompttechnique 1: baseline
            prompt_baseline = template_baseline.format(QUESTION_TEXT=q_text, CODE_DOMAIN=q_dom) + "\n" + response_format
            res_baseline = call_llm(prompt_baseline)

            # Prompttechnique 2: example
            print("Example: ")
            prompt_example = template_example.format(QUESTION_TEXT=q_text, CODE_DOMAIN=q_dom) + "\n" + response_format
            res_example = call_llm(prompt_example)
            
            # Prompttechnique 3: chain of thought
            print("CoT: ")
            prompt_chain_of_thought = template_chain_of_thought.format(
                QUESTION_TEXT=q_text, 
                CODE_DOMAIN=q_dom,
            ) + "\n" + response_format
            res_chain_of_thought = call_llm(prompt_chain_of_thought)

            # Store data
            results.append({
                'question_id': q_id,
                'iteration': i,                
                'pred_baseline': res_baseline,
                'pred_example': res_example,    
                'pred_chain_of_thought': res_chain_of_thought
            })

    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv('results/predictions.csv', index=False)
    print("Simulation complete. Results saved to results/predictions.csv")

if __name__ == "__main__":
    main()