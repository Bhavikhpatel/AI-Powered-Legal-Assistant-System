# Legal RAG on Bharatiya Nyaya Sanhita (BNS) with Neo4j and LLMs

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system to answer legal questions about the **Bharatiya Nyaya Sanhita (BNS)**. It parses the BNS PDF, automatically extracts legal concepts into a Neo4j knowledge graph, and uses a Large Language Model (LLM) to provide structured answers based on user queries.

[![Watch the demo on LinkedIn](./assests/coverPage.png)](https://www.linkedin.com/posts/bhavikhpatel_neo4j-python-llama-activity-7346329202394087425-23jN)

## 📜 Overview

The core challenge with legal documents is their dense, unstructured nature. This project tackles that by creating a structured, queryable knowledge base from the BNS text. When a user asks a question (e.g., by describing a scenario from a news article), the system intelligently finds the most relevant legal offense and provides a detailed, cited answer.

## ⚙️ How It Works

The system follows a two-phase process: **1. Indexing** and **2. Querying**.

### 1. Indexing Pipeline (One-time Setup)

1.  **PDF Loading & Chunking**: The `BNS.pdf` document is loaded and split into smaller, manageable text chunks using `langchain`.
2.  **Knowledge Extraction**: A powerful LLM (via the Groq API) reads through the text chunks and extracts key legal information into structured tuples of `(offence, chapter, section, punishment_clause)`.
3.  **Knowledge Graph Construction**: These structured tuples are used to populate a **Neo4j graph database**. Each offense, chapter, section, and punishment becomes a node, with relationships connecting them (e.g., `Offense -[:refersToSection]-> Section`).

### 2. Querying Pipeline (Real-time)

1.  **Semantic Search**: When a user enters a query, a `sentence-transformer` model converts it into a vector embedding. This embedding is used to find the most semantically similar `Offense` node in the Neo4j graph.
2.  **Context Retrieval**: Once the most relevant offense is identified, the system queries the Neo4j graph to retrieve all connected information (the chapter, section, and punishment details). This retrieved data serves as the "context".
3.  **Answer Generation**: The context and the original query are passed to the LLM with a carefully crafted prompt. The LLM then generates a final, human-readable, and structured legal explanation based *only* on the retrieved context.
4.  **Interactive UI**: The entire process is wrapped in a user-friendly **Gradio** web interface, which shows the final answer and the step-by-step processing logs.

## ✨ Features

-   **Automated Knowledge Extraction**: Automatically parses and structures complex legal text.
-   **Knowledge Graph Representation**: Models legal information in a highly connected and queryable Neo4j graph.
-   **Semantic Querying**: Understands the user's intent, not just keywords, to find relevant laws.
-   **Context-Aware Generation**: Generates answers grounded in the retrieved legal text, reducing hallucinations.
-   **Interactive Interface**: Simple and intuitive UI for easy interaction, complete with "under-the-hood" logs.

## 🛠️ Tech Stack

-   **LLM Orchestration**: LangChain
-   **LLM Inference**: Groq (for fast API access to models like Llama 3, Mixtral, etc.)
-   **Knowledge Graph**: Neo4j
-   **PDF Processing**: `PyPDFLoader`
-   **Semantic Search**: `sentence-transformers`
-   **Web UI**: Gradio

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.8+
-   A Neo4j instance (You can use a free [Neo4j AuraDB instance](https://neo4j.com/cloud/aura/) or a local Docker container).
-   A Groq API Key (Get one for free at [GroqCloud](https://console.groq.com/keys)).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <url>
    cd GraphRAG-on-Bharatiya-Nyaya-Sanhita-BNS
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You can create a `requirements.txt` file with the packages from the script: `langchain-community`, `pypdf`, `transformers`, `nltk`, `sentence_transformers`, `faiss-cpu`, `numpy`, `langchain_groq`, `gradio`, `neo4j`)*

### 3. Configuration

1.  **Place the PDF**: Download the Bharatiya Nyaya Sanhita (BNS) PDF and place it in the root directory of the project, naming it `BNS.pdf`.

2.  **Set Environment Variables**: Create a `.env` file in the root directory or set the environment variables directly. The script will need:
    -   `GROQ_API_KEY`: Your API key from Groq.
    -   `NEO4J_URI`: The URI for your Neo4j database (e.g., `neo4j+s://xxxx.databases.neo4j.io`).
    -   `NEO4J_USERNAME`: Your Neo4j username (e.g., `neo4j`).
    -   `NEO4J_PASSWORD`: Your Neo4j password.

3.  **Update the Script**: Modify the script to load these credentials securely. For example, in the `Graphclass`, update the `__init__` method:

    ```python
    import os
    from dotenv import load_dotenv

    load_dotenv()

    class Graphclass:
        def __init__(self, database="neo4j"):
            self.uri = os.getenv("NEO4J_URI")
            self.user = os.getenv("NEO4J_USERNAME")
            self.password = os.getenv("NEO4J_PASSWORD")
            # ... rest of the code
    ```
    Do the same for the `Inference` class to load the `GROQ_API_KEY`.

### 4. Usage

The script is designed to be run once for setup and then used for querying via the Gradio app.

1.  **Run the Indexing Pipeline (First Time Only)**:
    Uncomment the lines at the bottom of the script that perform the one-time setup:
    ```python
    # --- ONE-TIME SETUP ---
    # 1. Load and chunk the PDF
    pdf_handler = PDFFunctions()
    chunks = pdf_handler.pdf_to_chunks(["/BNS.pdf"])

    # 2. Extract structured tuples using the LLM
    inference_engine = Inference(api_key=os.getenv("GROQ_API_KEY"))
    tuples = inference_engine.extract_custom_tuples(chunks)
    inference_engine.save_tuples_to_file(tuples, "graphData.txt") # Optional: save for backup

    # 3. Build the Neo4j knowledge graph
    graph_builder = Graphclass()
    graph_builder.create_knowledge_graph(tuples)
    print("Knowledge graph has been successfully built.")
    # --- END OF SETUP ---
    ```
    Run the script once to populate your Neo4j database. After it completes, you can comment out these lines again.

2.  **Launch the Gradio App**:
    Make sure the setup lines are commented out, and then run the script. The Gradio application will start.
    ```bash
    python your_script_name.py
    ```
    Open the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`) to access the web interface.

## 💡 Future Improvements

- **Accuracy Optimization**: Enhance retrieval and reasoning precision by refining graph queries, embeddings, and prompt structure.

- **Highlight Graph Context**: Visually display the subgraph of retrieved legal nodes and relationships to help users verify and trust the source.

- **Feedback Loop**: Allow users to rate responses or suggest corrections, enabling continuous refinement of both graph queries and LLM behavior.


