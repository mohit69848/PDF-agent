# langgraph_workflow.py
from langgraph import Workflow, Node
from agent import PDFQAAgent

agent_instance = PDFQAAgent()

def ingest_node(state):
    pdf_path = state.get("pdf_path")
    count = agent_instance.ingest(pdf_path)
    return {"ingested_chunks": count}

def query_node(state):
    question = state.get("question", "").strip()
    if not question:
        return {"answer": "⚠️ No question provided."}

    # For Preface, Intro, Acknowledgments, prioritize first 15 pages
    front_pages = 15 if any(q in question.lower() for q in ["preface", "acknowledgments", "introduction"]) else 10
    agent_instance.chain = agent_instance.vs.get_retriever(k=5, front_pages=front_pages)
    
    res = agent_instance.ask_with_sources(question, k=5)
    return {"answer": res}

wf = Workflow(name="pdf-qa")
wf.add_node(Node("ingest", ingest_node))
wf.add_node(Node("query", query_node))
