from google.generativeai import GenerativeModel

class GeminiTutorAgent:
    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-flash")

    def ask(self, question: str, context: str) -> str:
        try:
            prompt = f"""
You are an AI tutor helping students understand concepts from study materials.

Context:
{context}

Question:
{question}

Answer in a simple and clear way:
"""
            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"❌ Error while processing: {str(e)}"
