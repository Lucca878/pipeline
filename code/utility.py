import pandas as pd  
import numpy as np  
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch 

#models and tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/DistilBERT')
model = AutoModelForSequenceClassification.from_pretrained('models/DistilBERT', use_safetensors=True)

def chatloop(frase):
    """
    Process the input statement using the model and return the prediction and confidence score.
    """
     # Tokenize the text and convert to input IDs
    inputs = tokenizer(frase, return_tensors="pt")

    # Get logits, predicted probabilities, and predicted label
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)  
    predicted_label = probabilities.argmax().item()
            
    # Get the class probability 
    class_prob = probabilities[0, predicted_label].item()
    return 1-predicted_label, class_prob*100

    """
    tokenize = tokenizer(frase, return_tensors='tf')
    for i in [tokenize]:
        h = model.generate(**i)
        decoded_pred = tokenizer.batch_decode(h, skip_special_tokens=True)
        h1 = model.generate(**i, return_dict_in_generate=True, output_scores=True)
        prob = np.max(np.exp(h1.scores) / np.sum(np.exp(h1.scores))) * 100  # Convert to percentage
    return decoded_pred[0], prob
    """

def load_statements():
    return pd.read_csv("data/hippocorpus_test_truncated.csv", sep=",")

