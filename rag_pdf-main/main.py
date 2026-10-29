import os
import google.generativeai as genai
from graph.tutor_flow import build_tutor_graph  

# Setup Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB3tcSYrrGlN55saJOOc5rBJ9D68BBYCQE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Build LangGraph flow
graph = build_tutor_graph()

# Chat loop
while True:
    q = input("\nAsk a question (or 'exit'): ")
    if q.lower() == 'exit':
        break

    # Use dictionary-based state
    state = {"question": q}
    final_state = graph.invoke(state)

    print("\n📘 Answer:", final_state["answer"])
