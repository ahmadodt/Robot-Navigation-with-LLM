from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import json
import random

class RobotPromptGenerator:
    def __init__(self, prompt_type, prompt_templates, map, goal, few_shots_file, num_few_shots):
        self.prompt_type = prompt_type
        self.prompt_templates = prompt_templates
        self.map = map
        self.goal = goal
        self.few_shots_file = few_shots_file
        self.num_few_shots = num_few_shots

        # Load few-shot examples
        with open(few_shots_file, 'r') as file:
            self.few_shots = json.load(file)

    def build_prompt(self) -> str:
        if self.prompt_type == "str":
            prompt = ""
            prompt += self.prompt_templates.get("main_prompt", "No main prompt available.")
            prompt = self.add_few_shots(prompt)
            prompt = self.check_surroundings(prompt)
            prompt = self.add_goal_command(prompt)
            return prompt

    def add_few_shots(self, prompt):
        # Select the desired number of few-shot examples
        selected_few_shots = random.sample(self.few_shots, min(self.num_few_shots, len(self.few_shots)))
        
        # Append each few-shot example to the prompt
        for shot in selected_few_shots:
            prompt += f"\nFew-shot Example: {shot['prompt']}\nExpected Result: {shot['expected_result']}\n"
        return prompt

    def check_surroundings(self, prompt):
        matrix = self.map
        rows = len(matrix)
        cols = len(matrix[0])

        # Finding the position of the robot (value 1)
        robot_position = None
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 1:
                    robot_position = (i, j)
                    break
            if robot_position:
                break

        if not robot_position:
            return "Robot not found."

        # Directions to check surrounding cells (8 directions)
        directions = {
            (-1, -1): "front_left",
            (-1, 0): "front",
            (-1, 1): "front_right",
            (0, -1): "left",
            (0, 1): "right",
            (1, -1): "back_left",
            (1, 0): "back",
            (1, 1): "back_right",
        }

        # Object types
        object_types = {
            2: "human_too_close",
            3: "electricity_box_close",
        }

        # Check each surrounding cell
        for direction, position in directions.items():
            new_row = robot_position[0] + direction[0]
            new_col = robot_position[1] + direction[1]

            # Check if the new position is within bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                cell_value = matrix[new_row][new_col]
                if cell_value in object_types:
                    template_key = f"{object_types[cell_value]}_{position}"
                    prompt += " " + self.prompt_templates.get(template_key, "")

        return prompt

    def add_goal_command(self, prompt):
        robot_position = self.find_robot_position()
        if not robot_position:
            return prompt + " Robot not found."

        # Calculate the difference between the goal and the robot position
        goal_row, goal_col = self.goal
        robot_row, robot_col = robot_position
        row_diff = goal_row - robot_row
        col_diff = goal_col - robot_col

        # Determine the direction and step count
        if row_diff < 0:
            row_direction = "front"
        elif row_diff > 0:
            row_direction = "back"
        else:
            row_direction = None

        if col_diff < 0:
            col_direction = "left"
        elif col_diff > 0:
            col_direction = "right"
        else:
            col_direction = None

        # Construct the command message
        steps = []
        if row_direction:
            steps.append(f"{abs(row_diff)} step{'s' if abs(row_diff) > 1 else ''} to the {row_direction}")
        if col_direction:
            steps.append(f"{abs(col_diff)} step{'s' if abs(col_diff) > 1 else ''} to the {col_direction}")

        if steps:
            command = " and ".join(steps)
            prompt += f" The goal is {command}."

        return prompt

    def find_robot_position(self):
        matrix = self.map
        rows = len(matrix)
        cols = len(matrix[0])

        # Finding the position of the robot (value 1)
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 1:
                    return (i, j)
        return None


prompt_templates = {
    #"main_prompt": "Given the following scenarios, generate a command for the robot to move towards the goal while considering any obstacles in its surroundings.",
    "main_prompt": "Depending on this situation give me a robot command: ",
    "human_too_close_front": "you are too close to a human in front of you, ",
    "human_too_close_back": "you are too close to a human behind you, ",
    "human_too_close_left": "you are too close to a human on your left, ",
    "human_too_close_right": "you are too close to a human on your right, ",
    "human_too_close_front_left": "you are too close to a human on your front left, ",
    "human_too_close_front_right": "you are too close to a human on your front right, ",
    "human_too_close_back_left": "you are too close to a human on your back left, ",
    "human_too_close_back_right": "you are too close to a human on your back right, ",
    "electricity_box_close_front": "you are too close to an electricity box in front of you, ",
    "electricity_box_close_back": "you are too close to an electricity box behind you, ",
    "electricity_box_close_left": "you are too close to an electricity box on your left, ",
    "electricity_box_close_right": "you are too close to an electricity box on your right, ",
    "electricity_box_close_front_left": "you are too close to an electricity box on your front left, ",
    "electricity_box_close_front_right": "you are too close to an electricity box on your front right, ",
    "electricity_box_close_back_left": "you are too close to an electricity box on your back left, ",
    "electricity_box_close_back_right": "you are too close to an electricity box on your back right, ",
    "end_prompt": "Which direction will you move to reach your goal and stay far from humans and electricity boxes?",
}

map_matrix = [
    [0, 0, 2, 0],
    [0, 1, 0, 3],
    [2, 0, 0, 0],
    [0, 0, 0, 0],
]

goal_position = (3, 2)

# Path to the fewshots.json file
few_shots_file = "few_shots.json"

robot_prompt_generator = RobotPromptGenerator(
    prompt_type="str",
    prompt_templates=prompt_templates,
    map=map_matrix,
    goal=goal_position,
    few_shots_file=few_shots_file,
    num_few_shots=0  # Set the desired number of few-shots to include
)

prompt = robot_prompt_generator.build_prompt()
print(prompt)

# Usage example

token = "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"
# Use the Llama 2-chat-hf model from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
model = AutoModelForCausalLM.from_pretrained(model_name,token=token)

def generate_command(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=100)
    command = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return command

print(generate_command(prompt))
