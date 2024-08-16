import json
import random

class RobotPromptGenerator:
    """Generates prompts for a robot based on its map and goal, and updates its position."""
    
    def __init__(self, prompt_type, prompt_templates, map_matrix, goal, few_shots_file, num_few_shots):
        """
        Initializes the RobotPromptGenerator with given parameters.

        Args:
            prompt_type (str): The type of prompt to generate.
            prompt_templates (str): Path to the prompt templates file.
            map_matrix (list): The current map matrix of the environment.
            goal (tuple): The goal position (row, col).
            few_shots_file (str): Path to the file containing few-shot examples.
            num_few_shots (int): Number of few-shot examples to include in the prompt.
        """
        self.prompt_type = prompt_type
        self.map_matrix = map_matrix
        self.goal = goal
        self.few_shots_file = few_shots_file
        self.num_few_shots = num_few_shots

        # Load few-shot examples
        self.few_shots = self._load_json(few_shots_file)
        
        # Load prompts
        self.prompt_templates = self._load_json(prompt_templates)
    
    def _load_json(self, file_path):
        """Load JSON data from a file."""
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def build_prompt(self) -> str:
        """Builds the prompt based on the specified prompt type."""
        if self.prompt_type == "str":
            prompt = self.prompt_templates.get("main_prompt", "No main prompt available.")
            prompt = self._add_few_shots(prompt)
            prompt = self._check_surroundings(prompt)
            prompt = self._add_goal_command(prompt)
            prompt += self.prompt_templates.get("end_prompt")
            return prompt
    
    def _add_few_shots(self, prompt: str) -> str:
        """Add few-shot examples to the prompt."""
        selected_few_shots = random.sample(self.few_shots, min(self.num_few_shots, len(self.few_shots)))
        for shot in selected_few_shots:
            prompt += f"\nFew-shot Example: {shot['prompt']}\nExpected Result: {shot['expected_result']}\n"
        return prompt

    def _check_surroundings(self, prompt: str) -> str:
        """Add information about the robot's surroundings to the prompt."""
        robot_position = self._find_robot_position()
        if not robot_position:
            return "Robot not found."

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

        object_types = {
            2: "human_too_close",
            3: "electricity_box_close",
        }

        for direction, position in directions.items():
            new_row = robot_position[0] + direction[0]
            new_col = robot_position[1] + direction[1]
            if 0 <= new_row < len(self.map_matrix) and 0 <= new_col < len(self.map_matrix[0]):
                cell_value = self.map_matrix[new_row][new_col]
                if cell_value in object_types:
                    template_key = f"{object_types[cell_value]}_{position}"
                    prompt += " " + self.prompt_templates.get(template_key, "")

        return prompt

    def _add_goal_command(self, prompt: str) -> str:
        """Add goal distance and direction to the prompt."""
        robot_position = self._find_robot_position()
        if not robot_position:
            return prompt + " Robot not found."

        goal_row, goal_col = self.goal
        robot_row, robot_col = robot_position
        row_diff = goal_row - robot_row
        col_diff = goal_col - robot_col

        steps = []
        if row_diff < 0:
            steps.append(f"{abs(row_diff)} step{'s' if abs(row_diff) > 1 else ''} to the front")
        elif row_diff > 0:
            steps.append(f"{abs(row_diff)} step{'s' if abs(row_diff) > 1 else ''} to the back")

        if col_diff < 0:
            steps.append(f"{abs(col_diff)} step{'s' if abs(col_diff) > 1 else ''} to the left")
        elif col_diff > 0:
            steps.append(f"{abs(col_diff)} step{'s' if abs(col_diff) > 1 else ''} to the right")

        if steps:
            command = " and ".join(steps)
            prompt += f" The goal is {command}."

        return prompt

    def _find_robot_position(self):
        """Find the robot's position in the map matrix."""
        for i, row in enumerate(self.map_matrix):
            for j, cell in enumerate(row):
                if cell == 1:
                    return (i, j)
        return None
    
    def update_map_with_command(self, command: str):
        """
        Update the map based on the command issued.

        Args:
            command (str): The command indicating the movement directions.

        Returns:
            list: Updated map matrix.
        """
        robot_position = self._find_robot_position()
        if not robot_position:
            return "Robot not found."

        row, col = robot_position
        direction_map = {
            "forward": (-1, 0),
            "backward": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        
        steps = [direction for direction in direction_map.keys() if direction in command.lower()]

        for direction in steps:
            move = direction_map[direction]
            row += move[0]
            col += move[1]

            if row < 0 or row >= len(self.map_matrix) or col < 0 or col >= len(self.map_matrix[0]):
                return "Movement out of bounds."

            self.map_matrix[robot_position[0]][robot_position[1]] = 0
            self.map_matrix[row][col] = 1
            robot_position = (row, col)

        return self.map_matrix
