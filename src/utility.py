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
    prompt = ""

    original_label = history.statement.classification
    original_prob = history.statement.score
    original_length = len(history.statement.statement.text_truncated.split())

    if original_label == 1:
        target_label = 'truthful'
    else:
        target_label = 'deceptive'

    if original_label == 1:
        statement_label = 'deceptive'
    else:
        statement_label = 'truthful'

    # Append historical data if any 
    if len(history.history_paraphrased) != 0:
        prompt += "Your previous attempt has failed, please try again. Here are your previous attempts:\n"
        label_map = {0: 'truthful', 1: 'deceptive'}
        for idx, history_paraphrased_statement in enumerate(history.history_paraphrased, 1):
            para_text = history_paraphrased_statement.paraphrased_text
            para_class = history_paraphrased_statement.classification
            para_score = history_paraphrased_statement.score
            # Map numeric to string if needed
            if isinstance(para_class, int):
                para_class = label_map[para_class]
            prompt += f"{idx}. {para_text}\n"
            prompt += f"   The AI evaluated your rewrite as {para_class} with a confidence of {para_score*100:.1f}%\n"
        prompt += f"You will now see the original statement and the instructions again:"

    # Append current statement
    prompt += f"An automated deception classifier predicted this statement to be {statement_label} with a confidence of {original_prob:.2f}%: {history.statement.statement.text_truncated} " 
    prompt += f"Rewrite this statement so that it appears {target_label} to the automated deception classifier." 
    prompt += f"In your rewrites, maintain the original statement’s meaning, ensure it is grammatically correct, and appears natural. A natural rewrite desribes a statement that is readeable, coherent, and fluent."
    prompt += f"Additionally, ensure your rewrite is within ±20 words of the length of the original statement ({original_length} words)."

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