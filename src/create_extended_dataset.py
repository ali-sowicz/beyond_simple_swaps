
import json
import random
import re


# 1. UTILITY FUNCTIONS (Basic I/O operations)
def load_json(input_path):
    """
    Load JSON data from file.
    
    Args:
        input_path (str): Path to the JSON file to load
        
    Returns:
        dict/list: Loaded JSON data, or None if error occurs
        
    Raises:
        Exception: If file cannot be loaded or parsed
    """
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    
def save_json(output_path, data):
    """
    Retrieve a specific question example by ID.
    
    Args:
        questions (dict): Questions dataset containing 'examples' list
        id (int): Unique identifier for the desired example
        
    Returns:
        str: Input text of the matching example, or None if not found
    """
    try:
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

# 2. DATA ACCESS FUNCTIONS (Retrieve specific data)
def get_set(sets, id, domain):
    """
    Retrieve object sets for a specific ID and domain.
    
    Args:
        sets (list): List of set configurations
        id (int): Example identifier
        domain (str): Domain category (e.g., 'animals', 'vehicles')
        
    Returns:
        dict: Set configuration containing objects and labels, or None if not found
    """
    for s in sets:
        if s["domain"] == domain and s["id"] == id:
            return s["sets"]

def get_example(questions, id):
    for example in questions["examples"]:
        if example["id"] == id:
            return example["input"]

# 3. CALCULATION FUNCTIONS (Pure logic, no side effects)
def how_many_steps(desired_num_of_steps):
    """
    Calculate number of step blocks needed for desired total steps.
    
    Args:
        desired_num_of_steps (int): Target number of reasoning steps
        
    Returns:
        int: Number of step blocks required
    """
    return int((desired_num_of_steps - 2)/3)
      
  
# 4. STEP GENERATION FUNCTIONS (Generate content)
def generate_extra_sentences(steps, category, num_blocks, i):
    """
    Generate additional no-action steps based on category and position.
    
    Args:
        steps (list): Available step templates
        category (str): Question category ('basic_swap', 'overlap', etc.)
        num_blocks (int): Number of step blocks to generate
        i (int): Current example index for position-based selection
        
    Returns:
        list: Generated sentences for insertion into questions
    """
    for item in steps:
        if item["category"] == "no_action":
            no_action_templates = item["templates"]
        if item["category"] == category:
            
            if category == "basic_swap":
                    for type_block in item["types"]:
                        type_id = type_block["type"]
                        if type_id == 1:
                            steps_dict = type_block["steps"]
                            return join_steps_blocks(steps_dict, num_blocks, no_action_templates = no_action_templates)
                            
            elif category == "overlap":
                    return get_overlap_steps(item, num_blocks, no_action_templates, i)
            elif category == "rules":
                    return get_rules_steps(item, num_blocks, no_action_templates, i)
            elif category == "negations":
                    return get_negations_steps(item, num_blocks, no_action_templates, i)
                                
def get_overlap_steps(item, num_blocks, no_action_templates, i):
    """
    Handle step selection for overlap category based on index position.
    
    Divides examples into 3 types based on position within 25-example blocks:
    - Type 1: positions 1-8
    - Type 2: positions 9-16  
    - Type 3: positions 17-25
    
    Args:
        item (dict): Step configuration for overlap category
        num_blocks (int): Number of blocks to generate
        no_action_templates (list): Templates for no-action steps
        i (int): Example index
        
    Returns:
        list: Selected and formatted step sentences
    """
    # Calculate relative position within 25-example blocks
    relative_i = ((i - 1) % 25) + 1  # Converts any index to 1-25 range
    
    # Determine which type to use based on relative position
    if 1 <= relative_i <= 8:
        type_id = 1
    elif 9 <= relative_i <= 16:
        type_id = 2
    else:  # 17-25
        type_id = 3

    # Find the matching type block
    for type_block in item["types"]:
        if type_block["type"] == type_id:
            steps_dict = type_block["steps"]
            return join_steps_blocks(steps_dict, num_blocks, no_action_templates)
    
    raise ValueError(f"No matching type {type_id} found for index {i}")

def get_rules_steps(item, num_blocks, no_action_templates, i):
    """Handle step selection for rules category based on index position."""
    # Calculate relative position within 25-example blocks
    relative_i = ((i - 1) % 25) + 1  # Converts any index to 1-25 range
    
    # Determine which type to use based on relative position
    if 1 <= relative_i <= 5:
        type_id = 1
    elif 6 <= relative_i <= 10:
        type_id = 2
    elif 11 <= relative_i <= 15:
        type_id = 3
    elif 16 <= relative_i <= 20:
        type_id = 4
    else:
        type_id = 5

    # Find the matching type block
    for type_block in item["types"]:
        if type_block["type"] == type_id:
            steps_dict = type_block["steps"]
            return join_steps_blocks(steps_dict, num_blocks, no_action_templates)
    
    raise ValueError(f"No matching type {type_id} found for index {i}")

def get_negations_steps(item, num_blocks, no_action_templates, i):
    """Handle step selection for negations category based on index position."""
    # Calculate relative position within 25-example blocks
    relative_i = ((i - 1) % 25) + 1  # Converts any index to 1-25 range
    
    # Determine which type to use based on relative position
    if 1 <= relative_i <= 9:
        type_id = 1
    elif 10 <= relative_i <= 18:
        type_id = 2
    else:
        type_id = 3

    # Find the matching type block
    for type_block in item["types"]:
        if type_block["type"] == type_id:
            steps_dict = type_block["steps"]
            return join_steps_blocks(steps_dict, num_blocks, no_action_templates)
    
    raise ValueError(f"No matching type {type_id} found for index {i}")

def join_steps_blocks(steps_dict, num_blocks, no_action_templates = None):
    """
    Combine step blocks into final sentence sequence.
    
    Args:
        steps_dict (dict): Dictionary of step blocks keyed by order
        num_blocks (int): Target number of blocks
        no_action_templates (list, optional): Templates for spacing between blocks
        
    Returns:
        list: Flattened list of sentences with proper ordering and spacing
    """
    # Convert string keys to int for sorting
    sorted_keys = sorted(steps_dict.keys(), key=int)
    ordered_steps = [steps_dict[key] for key in sorted_keys]

    # How many extra steps we need
    extra_needed = num_blocks - len(ordered_steps)

    # Add original steps first (in order)
    selected_blocks = ordered_steps.copy()

    # If more blocks are needed, randomly repeat some
    if extra_needed > 0:
        extras = random.choices(ordered_steps, k=extra_needed)
        random.shuffle(extras)  # Shuffle extras to vary the output
        selected_blocks += extras
    else:
        selected_blocks = ordered_steps[:num_blocks]

    # Flatten and join
    final_sentences = []
    for i, block in enumerate(selected_blocks):
        final_sentences.extend(block)
        # Inject a no-action sentence BETWEEN blocks if template list is provided
        if no_action_templates and i < len(selected_blocks) - 1:
            no_action = random.choice(no_action_templates)
            final_sentences.append(no_action)

    return final_sentences

# 5. FORMATTING FUNCTIONS (Text processing)
def format_numbered_sentences(sentences, start=2):
    """
    Format sentences with sequential numbering.
    
    Args:
        sentences (list): List of sentences to number
        start (int): Starting number for sequence (default: 2)
        
    Returns:
        tuple: (formatted_string, next_number)
            - formatted_string: Numbered sentences joined with newlines
            - next_number: Next available number for continuation
    """
    return "\n" + "\n".join(f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=start)), 2 + len(sentences)

def correct_numbered_second_half_sentences(second_half, new_length):
    """
    Adjust numbering in second half of questions after insertion.
    
    Args:
        second_half (list): Sentences from second half of original question
        new_length (int): New starting number for continuation
        
    Returns:
        list: Sentences with corrected numbering
    """
    # The first and second sentences are always numbered 2 and 3, we need to change the number to match the new length of steps
    unchanged_sentences = second_half[2:]
    corrected_sentences = []
    for i in range(0, 2):
        text = second_half[i]
        text = re.sub(r"\b\d+\.\s", f"\n{new_length + i}. " ,text)
        text = re.sub(r"\n\n", f"\n" ,text)
        corrected_sentences.append(text)
    return corrected_sentences + unchanged_sentences   

def list_to_sentences(sentences):
    return "\n".join(sentences) 

# 6. ASSEMBLY FUNCTIONS (Combine components)
def assemble_final_question(sentences, extra_sentences, items, insert_after):
    """
    Assemble complete question by inserting extra steps and filling templates.
    
    Args:
        sentences (list): Original question sentences
        extra_sentences (list): Additional no-action steps to insert
        items (dict): Object and label mappings for template filling
        insert_after (int): Position to insert extra sentences (default: after 3rd)
        
    Returns:
        str: Complete assembled question with proper numbering and filled templates
    """
    first_half = sentences[:insert_after]
    second_half = sentences[insert_after:]
    numbered_ext_sentences, new_length = format_numbered_sentences(extra_sentences)
    
    filled_ext_sentences = numbered_ext_sentences.replace("{{obj1}}", items["obj1"]).replace("{{obj2}}", items["obj2"]).replace("{{obj3}}", items["obj3"]).replace("{{place}}", items["place"]).replace("{{label1}}", items["label1"]).replace("{{label2}}", items["label2"]).replace("{{label3}}", items["label3"])

    second_half = correct_numbered_second_half_sentences(second_half, new_length)
    first = list_to_sentences(first_half)
    last = list_to_sentences(second_half)

    final_question = first + filled_ext_sentences + last
    return final_question

# 7. PROCESSING FUNCTIONS (High-level operations)
def update_questions_input(questions, final_question, id):
    for example in questions["examples"]:
        if example["id"] == id:
            example["input"] = final_question
    return questions

def process_question(questions, sets, domain, steps, category, num_blocks, insert_after=3):
    """
    Process all questions in dataset by adding no-action steps.
    
    Args:
        questions (dict): Original questions dataset
        sets (list): Object set configurations
        domain (str): Domain category
        steps (list): Available step templates
        category (str): Question category
        num_blocks (int): Number of step blocks to add
        insert_after (int): Position for step insertion
        
    Returns:
        dict: Modified questions dataset with added complexity
    """
    # We are inserting questions after 3rd index bcs, sentences [0], [1], and [2] are introduction + first step.
    for i in range (1,76):
        question = get_example(questions, i)
        items = get_set(sets, i, domain)
        sentences = question.split('\n')

        extra_sentences = generate_extra_sentences(steps, category, num_blocks, i)
        try:
            final_question = assemble_final_question(sentences, extra_sentences, items, insert_after)
            questions = update_questions_input(questions, final_question, i)
        except:
            print(f"---{i}-----PROBLEM WITH ID")
    return questions

# 8. MAIN EXECUTION
def run():
    """
    Main execution function for benchmark generation.
    
    Configures paths, loads data, processes questions with additional steps,
    and saves the enhanced benchmark dataset.
    
    Configuration:
        - Category: 'negations'
        - Domain: 'animals' 
        - Steps: 182 total reasoning steps
    """
    category = "negations"
    domain = "animals"
    input_path = f"/home/alaunix/thesis/data_joint/test/{category}/{domain}.json"
    output_path = f"/home/alaunix/thesis/data_big/test/{category}/{domain}.json"
    sets_path = f"/home/alaunix/thesis/data_joint/sets/objects_and_places_modified.json"
    steps_path = f"/home/alaunix/thesis/data_joint/sets/no_action_steps.json"
    # Keep this number even
    desired_num_of_steps = 182
    
    questions = load_json(input_path)
    sets = load_json(sets_path)
    steps = load_json(steps_path)

    num_blocks = how_many_steps(desired_num_of_steps)
    questions = process_question(questions, sets, domain, steps, category, num_blocks)
    
    save_json(output_path, questions)



if __name__ == '__main__':
    run()


    


    
