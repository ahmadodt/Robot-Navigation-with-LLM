from transformers import AutoModelForCausalLM, AutoTokenizer
from prompter import RobotPromptGenerator
from utils import load_config, initialize_model_and_tokenizer, generate_command, save_command, print_map

def main():
    """Main function to execute the prompt generation and map updating."""
    # Load configuration
    config = load_config('json_files/config.json')
    
    # Extract constants from config
    token = config["TOKEN"]
    model_name = config["MODEL_NAME"]
    few_shots_file = config["FEW_SHOTS_FILE"]
    prompt_templates = config["PROMPT_TEMPLATES"]
    command_output_file = config["COMMAND_OUTPUT_FILE"]
    map_matrix = config["MAP_MATRIX"]
    goal_position = tuple(config["GOAL_POSITION"])
    moves = config["MOVES"]
    num_few_shots = config["NUM_FEW_SHOTS"]
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_name, token)
    
    # Create the RobotPromptGenerator instance
    robot_prompt_generator = RobotPromptGenerator(
        prompt_type="str",
        prompt_templates=prompt_templates,
        map_matrix=map_matrix,
        goal=goal_position,
        few_shots_file=few_shots_file,
        num_few_shots=num_few_shots
    )
    
    print("Starting map:", robot_prompt_generator.map_matrix)
    goal_x = robot_prompt_generator.goal[0]
    goal_y = robot_prompt_generator.goal[1]
    for i in range(moves):

        #check if we reached the goal
        x,y = robot_prompt_generator._find_robot_position() 
        if x==goal_x and y == goal_y :
            if i==0: 
                print("NO moves needed we are alrready in the destination")
                break
            else:
                print(f"Only {i+1} moves where needed to reach the goal")
                break

        # Generate the prompt
        prompt = robot_prompt_generator.build_prompt()
        print("Prompt generated!")
        print("prompt: ",prompt )
        # Generate the command
        command = generate_command(prompt, model, tokenizer)
        print("command: ",command)
        # Update the map with the command
        updated_map = robot_prompt_generator.update_map_with_command(command)
        print(f"Map after move {i + 1}:")
        print_map(updated_map)

        # Save the command and updated map
        save_command(prompt, command, updated_map, command_output_file)
        

if __name__ == "__main__":
    main()
