import pandas as pd
from utility import load_statements, generate_prompt, feed_to_openAI, model_classification, reset_statement_count, store_data_csv, init_data_csv
from dao import history, statement, paraphrased_obj
import time
import datetime

# Total amount of statements that need to be tested
total_statement_count = 10

# How many times each statement will be tested maximum
max_statement_iteration = 10

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

# Reset statement count from test case to 0
reset_statement_count()
init_data_csv(timestamp)

for current_statement_index in range(0, total_statement_count):

    # Classify current statement with local model
    current_statement = load_statements()
    classification, probability = model_classification(current_statement.text_truncated)

    # Historical statement and paraphrased for the current statement
    # It stores the current statement and previous paraphrased sentences
    current_history = history(None, None)
    statement_instance = statement(current_statement, classification, probability)
    current_history.update_statement(statement_instance)

    for statement_iter in range(0, max_statement_iteration):
        # Generate prompt with historical statement and paraphrased
        prompt = generate_prompt(current_history)

        # Get current paraphrased statement from Chat GPT
        paraphrased_text = feed_to_openAI(prompt)

        # Classify current paraphrased statement with local model
        classification, probability = model_classification(paraphrased_text)
        paraphrased_instance = paraphrased_obj(paraphrased_text, classification, probability)

        current_history.add_paraphrased(paraphrased_instance)
        
        store_data_csv(timestamp, statement_instance, paraphrased_instance, prompt)

        # If the label is flipped, go to the next statement
        # 1 = false
        # 0 = true
        if classification == 1 and current_statement.condition == 'truthful':
            break
        if classification == 0 and current_statement.condition == 'deceptive':
            break


