# HealthGuard AI

An agentic healthcare information assistant that answers questions using local PDF references and trusted health websites. It is intended for educational use only and is not a substitute for professional medical advice, diagnosis, or treatment.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for creating the environment and installing the locked dependencies
- A Groq API key for language-model responses
- A Tavily API key for web-search features

## Setup

1. Clone the repository and open its directory:

	```bash
	git clone <repository-url>
	cd HealthCare_Agent
	```

2. Install the project dependencies:

	```bash
	uv sync
	```

3. Create a `.env` file in the project root and add your API keys:

	```dotenv
	GROQ_API_KEY=your_groq_api_key
	TAVILY_API_KEY=your_tavily_api_key
	```

	Get keys from [Groq](https://console.groq.com/keys) and [Tavily](https://app.tavily.com/home). Keep `.env` private; it is excluded from Git. Langfuse tracing is optional and can be configured with your Langfuse credentials if desired.

4. Add the PDF files you want the assistant to search to `knowledge-base/`. PDF files are not required to start the interface, but local-document answers need PDFs to be indexed.

## Run the app

From the project root, start Streamlit:

```bash
uv run streamlit run main.py
```

Open the local URL printed in the terminal (usually <http://localhost:8501>). In the sidebar, select **Ingest / update PDFs** to index files from `knowledge-base/` into the local `chroma_db/` vector database. The database is generated locally and is not tracked by Git, so each new clone needs to ingest its PDFs.

The first run may take longer while the sentence-transformer embedding model is downloaded. Keep the app running in the terminal; press `Ctrl+C` to stop it.

## RAG evaluation (optional)

After indexing PDFs, inspect retrieval scores:

```bash
uv run python rag_eval.py inspect
```

Evaluate the included test set:

```bash
uv run python rag_eval.py evaluate --testset rag_testset.json
```

To include the LLM-based evidence coverage check (uses API calls):

```bash
uv run python rag_eval.py evaluate --testset rag_testset.json --with-coverage
```

The included test set refers to specific PDF names and pages. Update `rag_testset.json` to match the documents you add to your own knowledge base.

## Troubleshooting

- **Missing API key:** Check that `.env` is in the project root and that the variable names are exactly `GROQ_API_KEY` and `TAVILY_API_KEY`.
- **No local PDF results:** Confirm that PDFs are in `knowledge-base/`, then click **Ingest / update PDFs** in the sidebar.
- **Dependency or Python version errors:** Confirm Python 3.11 or newer is available, then run `uv sync --no-install-project` again.
- **Port 8501 is busy:** Start the app on another port with `uv run streamlit run main.py --server.port 8502`.

## Medical-use notice

HealthGuard AI provides automated general information, not validated medical triage or personalized care. For urgent symptoms, contact local emergency services. Consult a qualified healthcare professional about medical concerns.
