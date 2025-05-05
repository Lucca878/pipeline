import pandas as pd
from utility import load_statements, generate_prompt, feed_to_openAI, model_classfication, reset_statement_count, store_data_csv, init_data_csv
from dao import history,statement,parapharsed_obj
import time
import datetime

# Total amount of statements need to be tested
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
    classfication, probablity = model_classfication(current_statement.text_truncated)

    # Historical statement and parapharesed for current statement
    # It stores current statement and previous paraphrased sentence
    current_history = history(None,None)
    statement_instance = statement(current_statement,classfication,probablity)
    current_history.update_statement(statement_instance)


    for statement_iter in range(0,max_statement_iteration):
        # Generate prompt with historical statment and parapharesd
        prompt = generate_prompt(current_history)

        # Get current parapharsed statement from Chat GPT
        paraphrased_text =  feed_to_openAI(prompt)

        # Classify current parapharsed statement with local model
        classfication, probablity = model_classfication(paraphrased_text)
        paraphrased_instance = parapharsed_obj(paraphrased_text,classfication,probablity)

        current_history.add_parapharsed(paraphrased_instance)
        
        store_data_csv(timestamp,statement_instance,paraphrased_instance,prompt)

        # If the label is flipped, go to next statement
        # 1 = false
        # 0 = ture
        if classfication == 1 and current_statement.condition == 'truthful':
            break
        if classfication == 0 and current_statement.condition == 'deceptive':
            break





