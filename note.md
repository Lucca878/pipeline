# old code 
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