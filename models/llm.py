from google import genai

class GeminiModel:
    def __init__(self, model_name: str, system_prompt: str):
        self.client = genai.Client()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def invoke(self, user_input: str, *, temperature=0.0, max_tokens=256) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                {
                    "role": "system",
                    "parts": [{"text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "parts": [{"text": user_input}]
                }
            ],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text.strip()

