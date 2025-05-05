import openai
import pandas as pd
from utility import chatloop, load_statements

# Initialize OpenAI client
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

def paraphrase_attack(statement, original_label, max_attempts=10):
    """
    Perform adversarial paraphrasing using GPT-4.1 to flip classification of a given statement.
    """
    target_label = "deceptive" if original_label == "truthful" else "truthful"

    original_prediction, original_prob = chatloop(statement)
    original_length = len(statement.split())

    conversation_history = [
        {"role": "system", "content": (
            f"This statement is {original_label} with a classifier confidence of {original_prob:.2f}%. "
            f"Rewrite this statement so that it appears {target_label} to an automated deception classifier. "
            f"Maintain the statement’s original meaning, grammatical correctness, and natural readability. "
            f"Ensure it is coherent, fluent, readable, and within ±20 words of the original length ({original_length} words)."
        )},
        {"role": "user", "content": statement}
    ]

    best_paraphrase, best_prob_diff, flipped = statement, 0, False

    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            temperature=0.7
        )

        paraphrase = response.choices[0].message.content.strip()
        paraphrase_length = len(paraphrase.split())

        # Ensure paraphrase length constraint
        if not (original_length - 20 <= paraphrase_length <= original_length + 20):
            continue

        paraphrase_prediction, paraphrase_prob = chatloop(paraphrase)
        feedback = f"Classifier prediction: {paraphrase_prediction}, Confidence: {paraphrase_prob:.2f}%"

        # Update conversation history with new paraphrase and feedback
        conversation_history.append({"role": "assistant", "content": paraphrase})
        conversation_history.append({"role": "user", "content": feedback})

        prob_diff = abs(paraphrase_prob - original_prob)

        # Keep paraphrase with the highest probability change
        if prob_diff > best_prob_diff:
            best_prob_diff = prob_diff
            best_paraphrase = paraphrase

        # Check if the classification flipped
        if paraphrase_prediction != original_prediction:
            flipped = True
            break

    return {
        "original": statement,
        "original_label": original_label,
        "paraphrase": best_paraphrase,
        "probability_change": best_prob_diff,
        "iterations": attempt + 1,
        "flipped": flipped
    }


# Process entire dataset
def run_full_paraphrase_attack():
    statements_df = load_statements()
    results = []

    for index, row in statements_df.iterrows():
        print(f"Processing statement {index + 1}/{len(statements_df)}...")
        result = paraphrase_attack(row['text_truncated'], row['condition'])
        results.append(result)

    # Save results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("paraphrased_statements_results.csv", index=False)

    print("Paraphrasing complete. Results saved to paraphrased_statements_results.csv.")


if __name__ == "__main__":
    run_full_paraphrase_attack()
