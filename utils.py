import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        return json.load(file)

def initialize_model_and_tokenizer(model_name, token):
    """Initialize the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    return model, tokenizer

def generate_command(prompt, model, tokenizer):
    """Generate a command based on the prompt using the model."""
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=400)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the command part from the generated text
    command_start_index = len(prompt)
    #remove the prompt from the output
    generated_text = generated_text[command_start_index:]
    # Splitting the string at the first period
    generated_text = generated_text.split('.', 1)[0]

    return generated_text

def save_command(prompt, command, map_matrix, file_path):
    """Save the prompt, command, and map matrix to a JSON file."""
    data_to_save = {
        "prompt": prompt,
        "command": command,
        "map_matrix": map_matrix
    }

    try:
        # Attempt to read existing data from the file
        with open(file_path, 'r') as json_file:
            try:
                existing_data = json.load(json_file)
            except json.JSONDecodeError:
                # If JSON is invalid, initialize as an empty list
                existing_data = []
    except FileNotFoundError:
        # If file does not exist, initialize as an empty list
        existing_data = []

    # Append new data
    existing_data.append(data_to_save)

    # Save updated data to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"Command and map saved to {file_path}")

def print_map(map_data):
    """just for better visual of the map matrix"""
    for row in map_data:
        print(' '.join(f'{cell:2}' for cell in row))

