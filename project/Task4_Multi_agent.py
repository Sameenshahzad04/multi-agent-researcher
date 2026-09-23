#%%
import os
import sys
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure 'project' is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config.setting import get_llm
from tools.calculator import calculate
from tools.search import web_search_tool
from tools.document_search import document_search
from services.vector_store import get_retriever



# ==========================================
# 1. Initialize Vector Store & LLM with Tools
# ==========================================
print("Checking and initializing vector store and embedding models...")
get_retriever()

llm = get_llm()

# 2. Gather Tools (Web Search, Calculator, Document Search)
tools = [web_search_tool, calculate, document_search]
# Bind tools to the LLM so it knows what tool schemas are available
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 1. Define Expanded AgentState
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    research_data: str          # The shared whiteboard for research notes
    analysis_feedback: str      # The analyst's critique or approval
    loop_count: int             # Safety counter to stop infinite loops


# ==========================================
# 2. Define Multi-Agent Nodes
# ==========================================

def researcher_node(state: AgentState):
    """The Researcher: Gathers info using the LLM and tools."""

    print("\n--- [RESEARCHER] Gathering facts... ---")
    messages = state["messages"]
    
    response = llm_with_tools.invoke(list(messages))
    
    # If the researcher calls a tool, we return the tool call message. 
    # If it writes text notes, we save it into 'research_data'.
    return {
        "messages": [response],
        "research_data": response.content if not response.tool_calls else "Gathering data via tools...",
        "loop_count": state.get("loop_count", 0) + 1
    }

def analyst_node(state: AgentState):
    """The Analyst: Checks research data for missing gaps or contradictions."""

    print("\n--- [ANALYST] Reviewing research findings... ---")
    research_content = state.get("research_data", "")
    
    prompt = f"""
    Analyze the following research data for completeness and missing details.
    - If information is missing or unclear, start your response with [NEED_MORE_INFO] followed by what is missing.
    - If the data is complete, start your response with [APPROVED].
    
    Data: {research_content}
    """
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return {
        "messages": [AIMessage(content=f"[Analyst Review]: {response.content}")],
        "analysis_feedback": response.content
    }

def writer_node(state: AgentState):
    """The Writer: Formats the verified research into a clean report."""

    print("\n--- [WRITER] Drafting final report... ---")
    research_content = state.get("research_data", "")
    
    prompt = f"Using the verified research data below, write a professional, structured final report:\n\n{research_content}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

# ==========================================
# 3. Define Conditional Routing Logic (The Loop)
# ==========================================
def route_after_analysis(state: AgentState):
    feedback = state.get("analysis_feedback", "")
    loop_count = state.get("loop_count", 0)
    
    if loop_count >= 3:
        print("--- [ROUTER] Max loops reached. Forcing Writer. ---")
        return "writer"
        
    if "[NEED_MORE_INFO]" in feedback:
        print("--- [ROUTER] Gaps found! Routing BACK to Researcher. ---")
        return "researcher"  # The feedback loop!
    else:
        print("--- [ROUTER] Approved! Routing to Writer. ---")
        return "writer"

#  5. Build and Compile the Graph app
# ==========================================
app = StateGraph(AgentState)

# Add all nodes
app.add_node("researcher", researcher_node)
app.add_node("tools", ToolNode(tools=tools)) # Handles tool execution for researcher
app.add_node("analyst", analyst_node)
app.add_node("writer", writer_node)

# Wiring the flow
app.add_edge(START, "researcher")

# If researcher calls a tool, go to tools node, else go to analyst
app.add_conditional_edges(
    "researcher",
    lambda state: "tools" if state["messages"][-1].tool_calls else "analyst",
    {"tools": "tools", "analyst": "analyst"}
)


# After tools run, go back to researcher to process tool output
app.add_edge("tools", "researcher")

# From analyst, use our loop router
app.add_conditional_edges(
    "analyst",
    route_after_analysis,
    {"researcher": "researcher", "writer": "writer"}
)

# Writer finishes the graph
app.add_edge("writer", END)




# Step 5.5: Add an in-memory checkpointer so state/history persists across turns
memory = MemorySaver()
app = app.compile(checkpointer=memory)
# app = app.compile()
# ==========================================
# 5. Interactive Session Runner
# ==========================================
def run_multi_agent_session():
    print("\n--- Multi-Agent Research System Initialized ---")
    thread_config = {"configurable": {"thread_id": "session-multi"}}

    while True:
        user_query = input("\nEnter your research topic (or 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
            
        if not user_query:
            continue

        # Initialize state with default values for our new keys
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "research_data": "",
            "analysis_feedback": "",
            "loop_count": 0
        }

        print("\nRunning Multi-Agent Workflow...")
        final_output = None
        
        for event in app.stream(initial_state, thread_config, stream_mode="values"):
            latest_message = event["messages"][-1]
            if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
                print(f"  ⚡ Action: Invoking tool -> {latest_message.tool_calls}")
            elif isinstance(latest_message, ToolMessage):
                print(f"  📦 Observation received from tool.")
            final_output = latest_message.content

        print(f"\nFinal Answer:\n{final_output}\n" + "=" * 60)

if __name__ == "__main__":
    run_multi_agent_session()