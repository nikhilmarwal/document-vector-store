from google import genai
from google.genai import types

class GeminiModel:
    def __init__(self, model_name: str, system_prompt: str):
        self.client = genai.Client()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def invoke(self, user_input: str, *, temperature=0.0, max_tokens=1024) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt, # Correct way to set system prompt
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
