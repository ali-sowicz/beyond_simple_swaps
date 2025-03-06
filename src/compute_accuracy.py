import os
import re
import json
from utils.config_loader import load_config

def extract_letter(text):
    """
    Extracts a single letter (A, B, or C) enclosed in parentheses.
    For example, it matches "(A)" but not "(some sentence)".
    """
    match = re.search(r"\(([ABC])\)", text)
    return match.group(1) if match else None

def extract_letter_from_response(response):
    """
    Splits the response by newline into non-empty lines and checks
    both the first and last line for a letter. If both yield a letter
    and they match, returns the letter with flag "correct_format".
    If they differ, returns the letter from the last line with flag
    "inconsistent_format". If a letter is only found in one of them,
    returns that letter with flag "correct_format". Otherwise, returns
    None with flag "wrong_format".
    """
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if not lines:
        return None, "wrong_format"
    
    letter_first = extract_letter(lines[0])
    letter_last = extract_letter(lines[-1])
    
    if letter_first and letter_last:
        if letter_first == letter_last:
            return letter_last, "correct_format"
        else:
            return letter_last, "inconsistent_format"
    if letter_last:
        return letter_last, "correct_format"
    if letter_first:
        return letter_first, "correct_format"
    
    return None, "wrong_format"

def compute_accuracy(json_dir):
    
    # Expected subfolders (you can also place this in config if needed)
    folders = ["overlap", "negations", "basic_swap", "rules"]
    results = {}
    evaluation_metadata = []  

    for folder in folders:
        folder_path = os.path.join(json_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        folder_results = {}

        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)

            correct = 0
            total = 0
            correct_format_count = 0

            for example in data:
                # Extract target letter (expected to be correctly formatted)
                target_letter = extract_letter(example["target"])
                extracted_letter, format_flag = extract_letter_from_response(example["response"])

                is_correct = (target_letter is not None and extracted_letter is not None and target_letter == extracted_letter)
                if is_correct:
                    correct += 1
                total += 1

                if format_flag == "correct_format":
                    correct_format_count += 1

                evaluation_metadata.append({
                    "id": example["id"],
                    "folder": folder,
                    "category": example.get("category", folder),
                    "file": json_file,
                    "target": target_letter,
                    "extracted": extracted_letter,
                    "extracted_by_hand": None,
                    "format": format_flag,
                    "is_correct": is_correct,
                    "response": example["response"]
                })

            accuracy = (correct / total) * 100 if total > 0 else 0
            folder_results[json_file] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "correct_format_count": correct_format_count
            }
            print(f"Processed {json_file} in {folder}: Accuracy = {accuracy:.2f}% (Total: {total}, Correct: {correct}, Correct Format: {correct_format_count})")

        results[folder] = folder_results

    # Save detailed evaluation metadata to a file in the json_dir
    metadata_file = os.path.join(json_dir, "evaluation_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(evaluation_metadata, f, indent=4)
    print(f"Evaluation metadata saved to {metadata_file}")

    # Write the accuracy summary to disk.
    accuracy_file = os.path.join(json_dir, "accuracy_summary.json")
    with open(accuracy_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Accuracy summary saved to {accuracy_file}")

    return results

if __name__ == "__main__":
    # Load configuration from config.yaml
    config = load_config()
    json_dir = config["compute_accuracy"]["json_dir"]
    accuracy_summary = compute_accuracy(json_dir)
    print("Accuracy Summary:")
    print(json.dumps(accuracy_summary, indent=4))
