# run_local.py
from agent import PDFQAAgent

if __name__ == "__main__":
    agent = PDFQAAgent()
    pdf_path = input("PDF path to ingest: ").strip()
    n = agent.ingest(pdf_path)
    print(f"✅ Ingested {n} chunks from PDF.\n")

    while True:
        q = input("Ask a question (or type 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break

        res = agent.answer(q, top_k=8)

        print("\n📌 Answer:\n")
        print(res["answer"])

        if res["sources"]:
            print("\n📖 Source snippets (top chunks):")
            for d in res["sources"]:
                page = d.metadata.get("page_number", "N/A")
                snippet = d.page_content[:200].replace("\n", " ")
                print(f"[Page {page}] {snippet}...\n")
