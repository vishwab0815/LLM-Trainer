import os
import json
from .llm_client import fetch_llm_response

class NLUEngine:
    def __init__(self, dataset_path: str = "dataset/intents.json"):
        self.dataset_path = dataset_path
        self.intents_data = self._load_dataset()
        self.intent_names = [i["name"] for i in self.intents_data.get("intents", [])]
        self.entity_types = list(self.intents_data.get("entities", {}).keys())

    def _load_dataset(self) -> dict:
        """Loads the intents dataset to build dynamic prompts."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_few_shot_prompt(self, user_input: str) -> str:
        """Dynamically constructs the LLM prompt using the dataset."""
        
        sys_prompt = f"""
You are an advanced Natural Language Understanding (NLU) system.
Your task is to classify the user's message into an intent and extract entities.

AVAILABLE INTENTS: {', '.join(self.intent_names)}
AVAILABLE ENTITY TYPES: {', '.join(self.entity_types)}

RULES:
1. ONLY predict an intent from the list. If it does not match, return "unknown".
2. Return STRICT JSON format only. Do not add markdown or conversational text.
3. Output schema:
{{
    "intent": "intent_name",
    "confidence": 0.95,
    "entities": {{
        "entity_type": "entity_value"
    }}
}}

EXAMPLES:
"""
        # Add very brief examples to save tokens for free tier
        for intent in self.intents_data.get("intents", []):
            intent_name = intent["name"]
            if intent["examples"]:
                example_text = intent["examples"][0]
                sys_prompt += f"User: {example_text}\nOutput:\n{{\"intent\": \"{intent_name}\", \"confidence\": 0.9, \"entities\": {{}}}}\n\n"

        sys_prompt += f"Now strictly classify the following message. Return ONLY JSON.\nUSER_MESSAGE: {user_input}\nOutput:\n"
        
        return sys_prompt

    def predict(self, user_input: str) -> dict:
        """
        Takes a raw user string, builds the prompt, fires it to the LLM, 
        and returns the structured JSON output.
        """
        prompt = self._build_few_shot_prompt(user_input)
        response_json = fetch_llm_response(prompt)
        return response_json
