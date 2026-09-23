# 🤖 Multi-Agent Research & Report Assistant

<img src="https://img.shields.io/badge/LangGraph-Stateful%20Agents-blue?style=for-the-badge&logo=python" alt="LangGraph">
<img src="https://img.shields.io/badge/OpenRouter-LLM%20Gateway-orange?style=for-the-badge" alt="OpenRouter">
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Search-green?style=for-the-badge" alt="ChromaDB">
<img src="https://img.shields.io/badge/SQLite-Long--Term%20Memory-purple?style=for-the-badge&logo=sqlite" alt="SQLite">

*A production-grade, multi-agent research pipeline that automates information gathering, fact-checking, and report generation with Human-in-the-Loop oversight.*

</div>

---

## ✨ Overview

Unlike standard chat-bots, this system operates as an **explicit state machine** using LangGraph. It mimics a real research team by splitting responsibilities across specialized agent roles, implementing a strict **RAG-first** search strategy, auditing its own outputs, and holding execution for human approval before final publication.

### 🌟 Key Production Features
* 🧠 **Multi-Agent Orchestration:** Coordinated roles for gathering (`Researcher`), auditing (`Analyst`), and writing (`Writer`).
* 🔍 **RAG-First Pipeline:** Prioritizes internal document corpora (`ChromaDB`) before falling back to live web search (`DuckDuckGo` / Tavily).
* 🛑 **Human-in-the-Loop (HITL):** Built-in LangGraph `interrupt` protocol allowing human review, editing, or rejection of draft reports.
* 💾 **Long-Term Memory (LTM):** SQLite checkpointer integration preserving state and historical session context.
* 🛡️ **Robust Error Handling:** Tool exception wrappers and message sanitization to prevent API 400 bad request crashes.

---

## 🏗️ System Architecture & Workflow


graph TD
    A[User Query] --> B[Researcher Node]
    B --> C{Tool Required?}
    C -->|Yes| D[Safe Tool Execution: RAG / Web / Calc]
    D --> B
    C -->|No| E[Analyst Node]
    E --> F{Gaps Found?}
    F -->|Yes: [NEED_MORE_INFO]| B
    F -->|No: [APPROVED]| G[Writer Node]
    G --> H[Human Review Interrupt]
    H -->|Revise| B
    H -->|Accept| I([Final Report Published])



---

## 📂 Project Tasks Breakdown

| Task | Core Focus | Key Technologies |
| --- | --- | --- |
| **Task 1** | ReAct-style base agent & tool integration | LangChain, OpenRouter |
| **Task 2** | Document ingestion, chunking, & vector retrieval | ChromaDB, Embeddings |
| **Task 3** | Explicit graph transition & state management | LangGraph `StateGraph`, `TypedDict` |
| **Task 4** | Multi-agent role splitting & feedback loop | Researcher $\leftrightarrow$ Analyst $\leftrightarrow$ Writer |
| **Task 5** | Production readiness, memory, & HITL | SQLite Checkpointer, LangGraph `interrupt` |
| **Task 6** | End-to-end evaluation, logging, & polish | Structured testing & error mitigation |

---

## 🚀 Quick Start Guide

### 1. Clone & Navigate

```bash
git clone [https://github.com/YOUR_USERNAME/multi-agent-researcher.git](https://github.com/YOUR_USERNAME/multi-agent-researcher.git)
cd multi-agent-researcher

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

```

### 3. Install Dependencies

Using `uv` (recommended) or `pip`:

```bash
uv pip install -e .
# or
pip install -r requirements.txt

```

### 4. Run the Pipeline

```bash
python project/Task5.py

```

---

## 💡 Example Usage Flow

1. **Input:** `"Analyze the latest developments regarding Apple iPhone releases."`
2. **Researcher Action:** Automatically checks internal documents via `document_search`, then queries web tools if data is missing.
3. **Analyst Audit:** Evaluates research data. If discrepancies are found, triggers a loop back to the researcher with instructions (`[NEED_MORE_INFO]`).
4. **Writer Synthesis:** Generates a clean markdown report.
5. **HITL Interruption:** Pauses terminal execution asking:
> *"Review the draft report below. Accept or provide revision instructions."*


6. **Output:** Type `accept` to finish or input specific critique for iteration.
