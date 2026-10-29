from langgraph.graph import StateGraph, END
from agents.retriever_agent import create_retriever_from_pdf
from agents.tutor_agent import GeminiTutorAgent

PDF_PATH = "/Users/karthikeyan/Desktop/ProJects/pdf_rag/data/55583main_vision_space_exploration2.pdf"
retriever = create_retriever_from_pdf(PDF_PATH)

# Node 1: retrieve context
def retrieve_node(state: dict):
    question = state.get("question")
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state  # Return as dictionary

# Node 2: tutor answers using context
def tutor_node(state: dict):
    question = state.get("question")
    context = state.get("context")
    tutor = GeminiTutorAgent()
    answer = tutor.ask(question, context)
    state["answer"] = answer
    return state  # Return as dictionary

# Build LangGraph flow
def build_tutor_graph():
    builder = StateGraph(dict)  # Specify that we want to use a dict type for state
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("tutor", tutor_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "tutor")
    builder.set_finish_point("tutor")

    return builder.compile()
