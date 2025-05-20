import pandas as pd  
import numpy as np  
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch 
from dao import history, statement, paraphrased_obj
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables

# Models and tokenizer
tokenizer = AutoTokenizer.from_pretrained('model')
model = AutoModelForSequenceClassification.from_pretrained('model', use_safetensors=True)

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def load_statements():
    with open("data/count.txt", "r+") as f:
        # Go to last line
        for line in f:
            pass
        last_line = line

        count = last_line
        count = int(float(count))
        count = count + 1
        f.write(str(count) + '\n')
        f.close()
        return pd.read_csv("data/hippocorpus_test_truncated.csv", sep=",").iloc[count]


def reset_statement_count():
    with open("data/count.txt", "w") as f:
        f.write('0' + '\n')
        f.close()
        

def generate_prompt(history):
    label_map = {0: 'truthful', 1: 'deceptive'}

    original_label = history.statement.classification
    original_prob = history.statement.score
    original_length = len(history.statement.statement['text_truncated'].split())
    original_text = history.statement.statement['text_truncated']

    # Map numeric to string if needed
    if isinstance(original_label, int):
        statement_label = label_map[original_label]
        target_label = label_map[1 - original_label]
    else:
        statement_label = original_label
        target_label = 'truthful' if original_label == 'deceptive' else 'deceptive'

    prompt = (
        "Your task is to rewrite a statement so that a machine learning model changes its prediction.\n"
        "You will see the original statement (taken from a previous study on truthful or deceptive autobiographical statements), "
        "the prediction of a machine learning model (whether the predicted label is truthful or deceptive) and the prediction confidence "
        "(i.e., the class probabilities of the statement belonging to the predicted class).\n"
        "You will also see any previous attempts (if there are any, as well as the prediction that resulted from these).\n"
        f"Original statement: {original_text}\n"
        f"Machine learning prediction (original statement): {statement_label}\n"
        f"Prediction confidence (original statement): {original_prob*100:.0f}%\n"
    )

    # Add previous attempts if any
    if len(history.history_paraphrased) > 0:
        prompt += f"\nPrevious attempts: {len(history.history_paraphrased)}\n"
        for idx, history_paraphrased_statement in enumerate(history.history_paraphrased, 1):
            para_text = history_paraphrased_statement.paraphrased_text
            para_class = history_paraphrased_statement.classification
            para_score = history_paraphrased_statement.score
            if isinstance(para_class, int):
                para_class = label_map[para_class]
            prompt += (
                f"\nStatement after attempt {idx}: {para_text}\n"
                f"Machine learning prediction (after attempt {idx}): {para_class}\n"
                f"Prediction confidence (after attempt {idx}): {para_score*100:.0f}%\n"
            )
    prompt += (
        f"\nNow rewrite the original statement (taking into account any previous attempts listed above) so that it appears more {target_label} to the machine learning classifier. "
        "Maintain the original statement’s meaning, ensure it is grammatically correct, and appears natural (i.e., that it is readable, coherent, and fluent). "
        f"Ensure that version is within ±20 words of the length of the original statement ({original_length} words).\n"
        "Your modification:"
    )
        

    return prompt


def feed_to_openAI(prompt):
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Only give me the requested paraphrase and nothing else."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def model_classification(paraphrased):
    # Tokenize the text and convert to input IDs
    inputs = tokenizer(paraphrased, return_tensors="pt")

    # Get logits, predicted probabilities, and predicted label
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)  
    predicted_label = probabilities.argmax().item()

    # 1 = false
    # 0 = true
    # Get the class probability 
    class_prob = probabilities[0, predicted_label].item()
    return predicted_label, class_prob


def init_data_csv(time):
    df = pd.DataFrame({ 'index': [],
                        'original_statement': [],
                        'os_class': [],
                        'os_score': [],
                        'para_statement': [],
                        'para_class': [],
                        'para_score': [],
                        'prompt': []})
    
    df.to_csv(f'data/exps/{time}_out.csv', mode='a', index=False, header=True)


def store_data_csv(time, statement_instance, paraphrased_instance, prompt):
    df = pd.DataFrame({ 'index': [statement_instance.statement['index']],
                        'original_statement': [statement_instance.statement['text_truncated']],
                        'os_class': [statement_instance.classification],
                        'os_score': [statement_instance.score],
                        'para_statement': [paraphrased_instance.paraphrased_text],
                        'para_class': [paraphrased_instance.classification],
                        'para_score': [paraphrased_instance.score],
                        'prompt': [prompt]})
    print(df)
    df.to_csv(f'data/exps/{time}_out.csv', mode='a', index=False, header=False)