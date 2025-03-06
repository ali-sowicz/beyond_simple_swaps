import os
import json
from collections import defaultdict
from utils.config_loader import load_config

def correct_and_evaluate_metadata(metadata_path, updated_metadata_path, accuracy_output_path):
    """
    Use when returned metadata from compute_accuracy.py had records with "wrong_format" or "inconsistent_format" 
    and you manually looked through it and corrected it by changing "extracted_by_hand" field from null to an answer.
    Loads evaluation metadata from metadata_path, applies corrections:
      - If "extracted" is None and "format" is "correct_format", change "format" to "wrong_format".
      - If "target" equals "extracted" or "extracted_by_hand", set "is_correct" to True.
    Then computes accuracy (percentage correct) and counts how often the format was "correct_format"
    for each file in each folder.
    
    Parameters:
      metadata_path (str): Path to the original evaluation_metadata.json file.
      updated_metadata_path (str): Path to save the updated metadata.
      accuracy_output_path (str): Path to save the accuracy summary as JSON.
    
    Returns:
      A tuple (updated_metadata, accuracy_summary)
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Apply corrections to each record
    for record in metadata:
        if record.get("extracted") is None and record.get("format") == "correct_format":
            record["format"] = "wrong_format"
        
        target = record.get("target")
        extracted = record.get("extracted")
        extracted_by_hand = record.get("extracted_by_hand")
        if (extracted is not None and target == extracted) or \
           (extracted_by_hand is not None and target == extracted_by_hand):
            record["is_correct"] = True

    # Compute summary per file per folder.
    summary = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0, "correct_format_count": 0}))
    
    for record in metadata:
        folder = record.get("folder")
        file_name = record.get("file")
        summary[folder][file_name]["total"] += 1
        if record.get("is_correct"):
            summary[folder][file_name]["correct"] += 1
        if record.get("format") == "correct_format":
            summary[folder][file_name]["correct_format_count"] += 1

    # Prepare the accuracy summary in the desired format.
    accuracy_summary = {}
    for folder, files in summary.items():
        accuracy_summary[folder] = {}
        for file_name, counts in files.items():
            total = counts["total"]
            correct = counts["correct"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            accuracy_summary[folder][file_name] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "correct_format_count": counts["correct_format_count"]
            }
    
    # Write updated metadata to disk.
    with open(updated_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Updated metadata saved to {updated_metadata_path}")

    # Write accuracy summary to disk.
    with open(accuracy_output_path, "w") as f:
        json.dump(accuracy_summary, f, indent=4)
    print(f"Accuracy summary saved to {accuracy_output_path}")
    
    return metadata, accuracy_summary

if __name__ == '__main__':
    # Load configuration from config.yaml
    config = load_config()
    metadata_path = config["manual_correction"]["metadata_path"]
    updated_metadata_path = config["manual_correction"]["updated_metadata_path"]
    accuracy_output_path = config["manual_correction"]["accuracy_output_path"]
    
    updated_metadata, accuracy_summary = correct_and_evaluate_metadata(
        metadata_path, updated_metadata_path, accuracy_output_path
    )
    
    print("Accuracy Summary:")
    print(json.dumps(accuracy_summary, indent=4))
