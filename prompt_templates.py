import json
import textwrap
from typing import List


class PromptTemplates:
    """Class for managing prompt templates for RiR tasks."""

    def __init__(self):
        self.prompt_dict = {}  # Dictionary to store prompt templates
        self.basic_prompt = None
        self.detailed_prompt = None

    def get_basic_prompt(self) -> str:
        """Returns the basic prompt template."""
        if self.basic_prompt is None:
            raise ValueError(
                "Basic prompt template is not set. Call load_prompt_template() to load it."
            )
        return self.basic_prompt

    def get_detailed_prompt(self) -> str:
        """Returns the detailed prompt template."""
        if self.detailed_prompt is None:
            raise ValueError(
                "Detailed prompt template is not set. Call load_prompt_template() to load it."
            )
        return self.detailed_prompt

    def load_evolved_prompt(self, prompt: str) -> None:
        """Loads an evolved prompt template.

        Args:
            prompt (str): Evolved prompt template to be loaded.
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        self.basic_prompt = prompt
        self.detailed_prompt = prompt

    def save_prompt_template(self, filename: str) -> None:
        """Saves the prompt templates to a JSON file.

        Args:
            filename (str): Path to the file where prompts will be saved.
        """
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string.")

        if self.basic_prompt is None or self.detailed_prompt is None:
            raise ValueError("Prompt templates are not set.")

        try:
            with open(filename, "w") as file:
                prompt_data = {
                    "basic_prompt": self.basic_prompt,
                    "detailed_prompt": self.detailed_prompt,
                }
                json.dump(prompt_data, file, indent=4)
        except IOError as e:
            raise IOError(f"Error saving prompt templates: {e}")

    def validate_prompt_format(self, prompt: str) -> None:
        """Validates the format of the prompt template.

        Raises an exception if the format is incorrect.

        Args:
            prompt (str): Prompt template to validate.
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        if not prompt.startswith(
            "Task: "
        ):  # Check if prompt starts with standard format
            raise ValueError("Prompt does not follow the expected format.")

    def load_prompt_template(self, filename: str) -> None:
        """Loads prompt templates from a JSON file.

        Args:
            filename (str): Path to the JSON file containing prompt templates.

        Raises:
            IOError: If the file cannot be opened or is not valid JSON.
        """
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string.")

        try:
            with open(filename, "r") as file:
                prompt_data = json.load(file)

            if (
                "basic_prompt" not in prompt_data
                or "detailed_prompt" not in prompt_data
            ):  # Check keys
                raise ValueError("Invalid prompt template JSON format.")

            self.basic_prompt = prompt_data["basic_prompt"]
            self.detailed_prompt = prompt_data["detailed_prompt"]

        except IOError as e:
            raise IOError(f"Error loading prompt templates: {e}") except json.JSONDecodeError as e:
            raise IOError(f"File is not a valid JSON: {e}")