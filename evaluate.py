import os
import json
import torch
import argparse
from tqdm import tqdm
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (
    MODEL_7B_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)

def load_dataset(dataset_path):
    """
    Load multiple choice questions dataset from JSON file.
    
    Args:
        dataset_path (str): Path to the JSON dataset file.
    
    Returns:
        list: List of question dictionaries.
    """
    with open(dataset_path, 'r') as f:
        return json.load(f)

def format_prompt(question, choices):
    """
    Format a multiple choice question into a prompt for the model.
    
    Args:
        question (str): The question text.
        choices (list): List of possible answer choices.
    
    Returns:
        str: Formatted prompt for model input.
    """
    prompt = f"Question: {question}\n\n"
    
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "\nPlease select the best answer from the choices above. Respond with the letter (A, B, C, or D) only."
    
    return prompt

def get_token_logits(model, prompt):
    """
    Get logits for the answer choices (A, B, C, D) from the model.
    
    Args:
        model (ChameleonInferenceModel): The model to query.
        prompt (str): The formatted prompt.
    
    Returns:
        torch.Tensor: Logits for answer choices [A, B, C, D]
    """
    batch_prompt_ui = [
        [
            {"type": "text", "value": prompt}
        ],
    ]
    
    options = Options()
    options.max_gen_len = 1  # We only need the first token of the response
    
    # Get the first token with its logits
    try:
        # Use step() to get the next token with its logits
        token = model.step(
            batch_prompt_ui=batch_prompt_ui,
            options=options
        )
        
        # Extract logits for A, B, C, D tokens
        letter_ids = [
            model.token_manager.tokenize_text("A")[0],
            model.token_manager.tokenize_text("B")[0],
            model.token_manager.tokenize_text("C")[0],
            model.token_manager.tokenize_text("D")[0]
        ]
        
        # Extract logits for these specific tokens
        letter_logits = torch.tensor([token.logits[0][id].item() for id in letter_ids])
        
        return letter_logits
        
    except Exception as e:
        print(f"Error getting logits: {e}")
        # Return equal probabilities as fallback
        return torch.tensor([0.25, 0.25, 0.25, 0.25])

def get_model_answer(model, prompt):
    """
    Get model's answer for a given prompt based on logits.
    
    Args:
        model (ChameleonInferenceModel): The model to query.
        prompt (str): The formatted prompt.
    
    Returns:
        str: The model's chosen answer letter (A, B, C, or D).
    """
    answer_tokens = ['A', 'B', 'C', 'D']
    logits = get_token_logits(model, prompt)
    
    # Find the token with the highest probability
    max_index = logits.argmax().item()
    return answer_tokens[max_index] 

def letter_to_index(letter):
    """
    Convert answer letter to answer index.
    
    Args:
        letter (str): Answer letter (A, B, C, or D).
    
    Returns:
        int: Corresponding index (0, 1, 2, or 3).
    """
    return ord(letter) - ord('A')

def evaluate_model(model, dataset):
    """
    Evaluate model on multiple choice questions.
    
    Args:
        model (ChameleonInferenceModel): The model to evaluate.
        dataset (list): List of question dictionaries.
    
    Returns:
        dict: Evaluation metrics.
    """
    correct = 0
    total = 0
    
    results = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        choices = item["choices"]
        correct_index = item["answer"]
        
        prompt = format_prompt(question, choices)
        answer_letter = get_model_answer(model, prompt)
        
        result = {
            "question": question,
            "choices": choices,
            "correct_answer": correct_index,
            "model_answer": answer_letter
        }
        
        answer_index = letter_to_index(answer_letter)
        result["model_answer_index"] = answer_index
        result["is_correct"] = (answer_index == correct_index)
        
        if answer_index == correct_index:
            correct += 1
            
        total += 1
        results.append(result)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "detailed_results": results
    }
    
    return metrics

def main(args):
    """
    Main function to run evaluation.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Load model
    print("Loading model...")
    model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path)
    
    # Run evaluation
    print(f"Evaluating model on {len(dataset)} questions...")
    metrics = evaluate_model(model, dataset)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    # Save results if output path is provided
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {args.output_path}")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate model on multiple choice questions.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="src/anole/evaluation/data/van_gogh.json",
        help="Path to the dataset JSON file."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="outputs/evaluation_results.json",
        help="Path to save evaluation results."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
