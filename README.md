# Multi-Agent Research & Report Assistant

An advanced, production-shaped agentic system built with **LangGraph**, **LangChain**, **OpenRouter**, **ChromaDB**, and **SQLite**. This system automates complex research workflows by orchestrating specialized agent roles, implementing strict RAG-first validation, and supporting Human-in-the-Loop (HITL) review.

---

## 🏗️ System Architecture & Workflow

The application operates as an explicit state machine using LangGraph across distinct specialized roles:
1. **Researcher:** Gathers facts using a mandatory **RAG-first** strategy (`document_search`), falling back to web search (`DuckDuckGo` / Tavily) only when necessary.
2. **Analyst:** Audits the gathered facts, independently verifies information, and flags gaps or contradictions. If information is missing, it commands the researcher to loop back with precise instructions (`[NEED_MORE_INFO]`).
3. **Writer:** Compiles verified research notes into a professional, structured markdown report.
4. **Human-in-the-Loop (HITL):** Pauses execution before final output using LangGraph `interrupt`, allowing a human reviewer to **accept** or request **revisions** with custom feedback.

🛠️ Project Tasks BreakdownTask 1: ReAct-style base agent setup with tool-binding and error-safe wrappers.Task 2: Document ingestion and RAG pipeline using ChromaDB and vector retrieval.Task 3: Transitioning black-box agent logic into an explicit LangGraph state machine.Task 4: Multi-agent orchestration (Researcher $\leftrightarrow$ Analyst $\leftrightarrow$ Writer loop).Task 5: Production readiness with SQLite persistent checkpointers (Long-Term Memory) and HITL interrupts.Task 6: Comprehensive logging, error handling, and robust session execution.🚀 Getting Started1. Clone the RepositoryBashgit clone https://github.com/YOUR_USERNAME/multi-agent-researcher.git
cd multi-agent-researcher
2. Set Up Environment VariablesCreate a .env file in the root directory and add your API keys:Code snippetOPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
3. Install DependenciesUsing uv (recommended) or pip:Bashuv pip install -e .
# or
pip install -r requirements.txt
4. Run the ApplicationBashpython project/Task5.py
