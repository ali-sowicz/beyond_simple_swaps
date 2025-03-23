import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from utils.config_loader import load_config

config = load_config()
login(token=config["models"]["huggingface_token"])
gpu_id = config["run_experiment"].get("gpu_id", 0)  # Default to GPU 0 if not specified

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    model.to(device)

    return model, tokenizer

def infer(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def generate_response(model, tokenizer, prompts):
    """Generate responses for a batch of prompts."""
    if isinstance(prompts, str):  # If a single string is passed, convert it into a list
        prompts = [prompts]

    messages_batch = [
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    text_batch = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4024,
    )
    # Remove input prompt tokens from generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # Process each response to extract the final answer
    final_answers = []
    for response in responses:
      final_answers.append(response)

    return final_answers  # Returns a list of responses


def load_few_shot_examples(test_dir, category_name):
    """
    Load few-shot examples from the corresponding dev directory.
    It replaces '/test/' with '/dev/' in the given test_dir.
    """
    dev_dir = test_dir.replace("/test/", "/dev/")
    few_shot_file = os.path.join(dev_dir, f"{category_name}.json")
    if os.path.exists(few_shot_file):
        with open(few_shot_file, "r") as f:
            data = json.load(f)
            return data.get("examples", [])
    return None

def generate_prompt(prompt_type, example_input, few_shot_examples=None):
    if prompt_type == "prompt_without_cot":
        return f"Answer this multiple-choice question without carrying out chain-of-thought prompting. Do not figure this out step by step. Give your answer immediately, Answer in format '(letter)'. Return final position with minimal analysis. {example_input}"
    elif prompt_type == "prompt_with_cot":
        return f"Provide the rationale before answering, and then give the answer in format '(letter)': {example_input}"
    elif prompt_type == "prompt_few_shot" and few_shot_examples:
        prefix = "Following are some examples of Input and Answers. "
        few_shot_prompt = "\n".join(
            [f"Input: {ex['input']}\nAnswer: {ex['target']}" for ex in few_shot_examples]
        )
        suffix = "Answer this multiple-choice question without carrying out chain-of-thought prompting. Do not figure this out step by step. Give your answer immediately, Answer in format '(letter)'. Return final position with minimal analysis. "
        return f"{prefix}{few_shot_prompt}\n{suffix}{example_input}"
    return example_input

def process_dataset(json_dirs, output_dir, prompt_type):
    os.makedirs(output_dir, exist_ok=True)
    MODEL_NAMES = config["models"]["names"]
    
    for model_name in MODEL_NAMES:
        print(f"Processing model: {model_name}")
        model, tokenizer = load_model(model_name)
        
        for json_dir in json_dirs:
            results = []
            # Use basename to get the folder name (e.g., "basic_swap")
            type_q = os.path.basename(json_dir)
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            
            # Create subdirectory for the folder inside output_dir
            type_q_output_dir = os.path.join(output_dir, type_q)
            os.makedirs(type_q_output_dir, exist_ok=True)
            
            for json_file in json_files:
                category_name = json_file[:-5]  # Remove ".json" to get the category
                few_shot_examples = (
                    load_few_shot_examples(json_dir, category_name)
                    if prompt_type == "prompt_few_shot"
                    else None
                )
                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)
                
                for example in data.get("examples", []):
                    prompt = generate_prompt(prompt_type, example["input"], few_shot_examples)
                    response = generate_response(model, tokenizer, prompt)
                    results.append({
                        "id": example["id"],
                        "category": category_name,
                        "model": model_name,
                        "input": example["input"],
                        "target": example["target"],
                        "response": response
                    })
            
            output_file = os.path.join(
                type_q_output_dir, f"{model_name.replace('/', '_')}_{prompt_type}_output.json"
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")

def process_dataset(json_dirs, output_dir, prompt_type="prompt_with_cot", batch_size=1):
    os.makedirs(output_dir, exist_ok=True)

    MODEL_NAMES = config["models"]["names"]


    for model_name in MODEL_NAMES:
        print(f"Processing model: {model_name}")
        model, tokenizer = load_model(model_name)

        for json_dir in json_dirs:
            results = []
            type_q = json_dir.split('/')[-1]
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

            # Create subdirectory for type_q inside output_dir
            type_q_output_dir = os.path.join(output_dir, type_q)
            os.makedirs(type_q_output_dir, exist_ok=True)

            for json_file in json_files:
                category_name = json_file[:-5]  # Remove .json extension
                few_shot_examples = (
                    load_few_shot_examples(json_dir, category_name)
                    if prompt_type == "prompt_few_shot"
                    else None
                )

                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)

                examples = data.get("examples", [])

                # **Process in Batches**
                for i in range(0, len(examples), batch_size):
                    batch = examples[i:i + batch_size]

                    # Generate batch prompts
                    batch_prompts = [generate_prompt(prompt_type, example["input"], few_shot_examples) for example in batch]

                    # Generate batch responses
                    batch_responses = generate_response(model, tokenizer, batch_prompts)  # Ensure generate_response can handle multiple inputs

                    # Store results
                    for example, response, prompt in zip(batch, batch_responses, batch_prompts):
                        results.append({
                            "id": example["id"],
                            "category": category_name,
                            "model": model_name,
                            "input": example["input"],
                            "prompt": prompt,
                            "target": example["target"],
                            "response": response
                        })

            output_file = os.path.join(type_q_output_dir, f"{model_name.replace('/', '_')}_{prompt_type}_output.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")


if __name__ == '__main__':
    # Retrieve run_experiment configuration from the YAML file.
    run_config = config["run_experiment"]
    batch_size = run_config["batch_size"]
    dataset_dir = run_config["dataset_dir"]
    output_dir = run_config["output_dir"]
    prompt_type = run_config["prompt_type"]
    subfolders = run_config["subfolders"]
    
    json_dirs = [os.path.join(dataset_dir, subfolder) for subfolder in subfolders]

    process_dataset(json_dirs, os.path.join(output_dir, prompt_type), prompt_type, batch_size)
