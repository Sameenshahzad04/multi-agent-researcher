import os
import sys
import warnings
import sqlite3
from typing import Annotated, Sequence, TypedDict
import uuid

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure 'project' is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from project.config.setting import get_llm
from project.tools.calculator import calculate
from project.tools.search import web_search_tool
from project.tools.document_search import document_search
from project.services.vector_store import get_retriever

# =============================================================================
# 1. Initialization & Environment
# =============================================================================
print("Initializing vector store and LLM...")
get_retriever()

llm = get_llm()

# =============================================================================
# 2. Corrected Safe Tool Wrapper (LangChain Compatible)
# =============================================================================
def safe_tool_wrapper(tool_obj):
    """
    Safely wraps a LangChain BaseTool / StructuredTool or regular function.
    Catches execution errors gracefully so the agent can self-correct.
    """
    # If it's a LangChain tool, we can use its .invoke() method safely
    original_func = getattr(tool_obj, "func", None)
    tool_name = getattr(tool_obj, "name", getattr(tool_obj, "__name__", "tool"))
    
    def wrapper(*args, **kwargs):
        try:
            # If it's a LangChain tool object, call its invoke method or underlying func
            if hasattr(tool_obj, "invoke"):
                return tool_obj.invoke(*args, **kwargs)
            elif original_func:
                return original_func(*args, **kwargs)
            else:
                return tool_obj(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ Tool Error Caught in {tool_name}: {str(e)}")
            return f"[ERROR] Tool failed due to exception: {str(e)}. Try an alternative tool or search query."

    # Preserve metadata or return the tool if wrapping fails at definition time
    return tool_obj

# Wrap tools safely
wrapped_web_search = safe_tool_wrapper(web_search_tool)
wrapped_calculate = safe_tool_wrapper(calculate)
wrapped_document_search = safe_tool_wrapper(document_search)

tools = [wrapped_web_search, wrapped_calculate, wrapped_document_search]
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# 3. Long-Term Memory (SQLite Checkpointer)
# =============================================================================
db_path = "research_sessions.sqlite"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory_checkpointer = SqliteSaver(conn)

# =============================================================================
# 4. Agent State Schema
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    research_data: Annotated[str, lambda x, y: f"{x}\n\n{y}" if x and y else (y or x)]      # Shared workspace text for facts
    analysis_feedback: str      # Analyst evaluation notes
    loop_count: int             # Safety counter
    review_status: str          # Tracks 'accept' or 'revise'
    current_agent: str

# 5. Helper: Sanitize Messages for LLM Providers (Prevents 400 Errors)
# =============================================================================
def sanitize_messages_for_llm(messages):
    """Ensures no message has empty content and no tool calls, preventing API 400 errors."""
    cleaned = []
    for msg in messages:
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        
        if not content and not tool_calls:
            if isinstance(msg, AIMessage):
                cleaned.append(AIMessage(content="[System note: previous step completed]"))
            elif isinstance(msg, ToolMessage):
                cleaned.append(ToolMessage(content="[Tool executed successfully]", tool_call_id=getattr(msg, "tool_call_id", "default")))
            else:
                cleaned.append(HumanMessage(content="[User action]"))
        else:
            cleaned.append(msg)
    return cleaned
# =============================================================================
# 5. Agent Nodes
# =============================================================================
RESEARCHER_PROMPT = (
    "You are an advanced research assistant.\n"
    "1. **MANDATORY RAG-FIRST**: You MUST call `document_search` first for every user query before resorting to web search.\n"
    "2. Only use `web_search` if the internal corpus (`document_search`) lacks sufficient details.\n"
    "3. **Never use search operators like 'site:'** in tool calls.\n"
    "4. If you receive human revision feedback, address the feedback explicitly."
)


def researcher_node(state: AgentState):
    print("\n--- [RESEARCHER] Gathering and synthesizing facts... ---")
    messages = state["messages"]
    full_messages = [{"role": "system", "content": RESEARCHER_PROMPT}] + list(messages)
    
    try:
        response = llm_with_tools.invoke(full_messages)
    except Exception as e:
        response = AIMessage(content=f"LLM Error: {str(e)}")

# 🔍 Print what the researcher is saying or doing
    if response.content:
        print(f"💬 [Researcher Message]: {response.content}")
    if response.tool_calls:
        print(f"⚡ [Researcher Tool Request]: {response.tool_calls}")
    else:
        print(f"📝 [Researcher Findings Synthesized]: {response.content[:300]}..." if response.content else "📝 [Researcher]: No direct text output.")
    # Only save actual content as research data. If it's just calling a tool, don't wipe existing notes.
    new_research = response.content if (response.content and not response.tool_calls) else ""

    return {
        "messages": [response],
        "research_data": new_research,
        "loop_count": state.get("loop_count", 0) + 1,
        "current_agent": "researcher"
    }

# def researcher_node(state: AgentState):
#     print("\n--- [RESEARCHER] Gathering and synthesizing facts... ---")
#     messages = state["messages"]
#     full_messages = [{"role": "system", "content": RESEARCHER_PROMPT}] + list(messages)
    
#     try:
#         response = llm_with_tools.invoke(full_messages)
#     except Exception as e:
#         response = AIMessage(content=f"LLM Error: {str(e)}")

#     return {
#         "messages": [response],
#         "research_data": response.content if not response.tool_calls else "Gathering data via tools...",
#         "loop_count": state.get("loop_count", 0) + 1,
#         "current_agent": "researcher" 
#     }

# def analyst_node(state: AgentState):
#     print("\n--- [ANALYST] Auditing findings and verifying facts with tools... ---")
#     research_content = state.get("research_data", "")
    
#     prompt = f"""
#     Review the research data. Ensure internal document corpus (RAG) via `document_search` was checked.
#     You have access to tools (`document_search`, `web_search_tool`, `calculate`). Use them to independently verify claims (e.g., checking if stated latest models or facts are accurate).
#     - If info is completely missing or RAG was skipped when relevant files exist, start with [NEED_MORE_INFO] followed by what is needed.
#     - If sufficient information is present, start with [APPROVED].
    
#     Data: {research_content}
#     """
    
#     response = llm_with_tools.invoke([{"role": "user", "content": prompt}])
    
#     updates = {
#         "messages": [response],
#         "current_agent": "analyst" 
#     }
#     if not response.tool_calls:
#         updates["analysis_feedback"] = response.content
#     return updates

# def writer_node(state: AgentState):
#     print("\n--- [WRITER] Compiling draft report... ---")
#     research_content = state.get("research_data", "")
    
#     prompt = f"Using the verified research notes below, write a professional report:\n\n{research_content}"
#     response = llm.invoke([{"role": "user", "content": prompt}])
    
#     return {
#         "messages": [AIMessage(content=response.content)]
#     }

def analyst_node(state: AgentState):
    print("\n--- [ANALYST] Auditing findings and verifying facts with tools... ---")
    research_content = state.get("research_data", "")
    messages = state["messages"]
    safe_messages = sanitize_messages_for_llm(messages)

    # Safely extract the latest user query
    latest_user_query = "New Research Request"
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_query = msg.content
            break
    
    prompt = f"""
    CRITICAL FOCUS: You must evaluate the research strictly against THIS LATEST user request: "{latest_user_query}". 
    Ignore any previous unrelated topics or historical chat queries.
    
    Review the current research data. Ensure internal document corpus (RAG) via `document_search` was checked.
    You have access to tools (`document_search`, `web_search_tool`, `calculate`). Use them to independently verify claims.
    
    - If information is missing, incomplete, or if you found alternative/newer facts that the researcher missed, start your response with [NEED_MORE_INFO] followed by clear, actionable instructions on what the researcher needs to search for or fix regarding "{latest_user_query}".
    - If sufficient information is present and accurate, start with [APPROVED].
    
    Current Research Data:
    {research_content}
    """
    
    try:
        response = llm_with_tools.invoke(safe_messages + [{"role": "user", "content": prompt}])
    except Exception as e:
        response = AIMessage(content=f"LLM Error: {str(e)}")

    if response.content:
        print(f"💬 [Analyst Message]: {response.content}")
    if response.tool_calls:
        print(f"⚡ [Analyst Tool Request]: {response.tool_calls}")

    updates = {
        "messages": [response],
        "current_agent": "analyst"
    }
    if not response.tool_calls:
        updates["analysis_feedback"] = response.content
    return updates


def writer_node(state: AgentState):
    print("\n--- [WRITER] Compiling draft report... ---")
    research_content = state.get("research_data", "")
    messages = state.get("messages", [])
    
    if len(research_content.strip()) < 20:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and "User Safety" not in msg.content:
                research_content = msg.content
                break

    prompt = f"""
    You are a professional technical writer. Using the verified research notes below, write a comprehensive, well-structured professional report:

    {research_content}
    """
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }


def human_review_node(state: AgentState):
    print("\n--- [SYSTEM] Pausing for Human Review ---")
    latest_draft = state["messages"][-1].content
    
    human_decision = interrupt({
        "question": "Review the draft report below. Accept or provide revision instructions.",
        "draft_report": latest_draft
    })
    
    status = human_decision.get("status", "accept")
    feedback = human_decision.get("feedback", "")
    
    if status == "revise":
        print(f"   Human requested revision: {feedback}")
        return {
            "review_status": "revise",
            "messages": [HumanMessage(content=f"Revision Instruction from Human: {feedback}")]
        }
    
    print("   Human accepted the draft!")
    return {"review_status": "accept"}

# =============================================================================
# 6. Conditional Routing
# =============================================================================
def route_after_analysis(state: AgentState):
    feedback = state.get("analysis_feedback", "")
    if state.get("loop_count", 0) >= 5:
        print("--- [ROUTER] Max loops reached. Forcing Writer. ---")
        return "writer"
    if "[NEED_MORE_INFO]" in feedback:
        print("--- [ROUTER] Gaps found! Routing BACK to Researcher. ---")
        return "researcher"
    print("--- [ROUTER] Approved! Routing to Writer. ---")
    return "writer"

def route_after_human_review(state: AgentState):
    status = state.get("review_status", "accept")
    if status == "revise":
        print("--- [ROUTER] Revision requested. Looping back to Researcher! ---")
        return "researcher"
    return END

# =============================================================================
# 7. Build Graph Workflow
# =============================================================================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools=tools))
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("human_review", human_review_node)

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges(
    "researcher",
    lambda state: "tools" if state["messages"][-1].tool_calls else "analyst",
    {"tools": "tools", "analyst": "analyst"}
)
# workflow.add_edge("tools", "researcher")

# If analyst calls a tool, route to tools; otherwise run route_after_analysis
workflow.add_conditional_edges(
    "analyst",
    lambda state: "tools" if state["messages"][-1].tool_calls else route_after_analysis(state),
    {"tools": "tools", "researcher": "researcher", "writer": "writer"}
)

workflow.add_conditional_edges(
    "tools",
    lambda state: state.get("current_agent", "researcher"),
    {"researcher": "researcher", "analyst": "analyst"}
)

workflow.add_edge("writer", "human_review")
workflow.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {"researcher": "researcher", END: END}
)

app = workflow.compile(checkpointer=memory_checkpointer)



def run_task5_session():
    print("\n" + "="*60)
    print("🚀 Task 5: Production Graph (HITL + LTM)")
    print("="*60)
    # Generate a unique thread ID for this session so old topics don't bleed in
    session_id = str(uuid.uuid4())[:8]
    thread_config = {"configurable": {"thread_id": f"session-{session_id}"}}
    print(f"🔒 Active Thread ID: session-{session_id}")
    while True:
        user_query = input("\nEnter your research request (or 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        # Initial state or resume command
        current_input = {
            "messages": [HumanMessage(content=user_query)],
            "research_data": "",
            "analysis_feedback": "",
            "loop_count": 0,
            "review_status": "accept",
            "current_agent": "researcher"
        }

        while True:
            print("\n⏳ Running workflow...")
            events = app.stream(current_input, thread_config, stream_mode="values")
            
            review_packet = None
            final_output = ""

            # Stream the events until it finishes or hits an interrupt
            for event in events:
                if "messages" in event and event["messages"]:
                    latest_message = event["messages"][-1]
                    final_output = latest_message.content
                    if isinstance(latest_message, AIMessage) and latest_message.tool_calls:

                        if latest_message.tool_calls:
                            print(f"  ⚡ Action: Invoking tool -> {latest_message.tool_calls}")
                        elif latest_message.content:
                            print(f"  💬 AI Output: {latest_message.content[:250]}...")
                    elif isinstance(latest_message, ToolMessage):
                        print(f"   📦 Observation received from tool.")

                if "__interrupt__" in event:
                    review_packet = event["__interrupt__"]

            # If no interrupt was hit, the graph reached the END! Break out of the review loop.
            if not review_packet:
                print(f"\n✅ Final Result:\n{final_output}\n" + "="*60)
                break

            # If an interrupt WAS hit, show the draft and get human feedback
            print("\n" + "~"*50)
            print("🛑 INTERRUPT: Review Draft Report")
            print("~"*50)
            for item in review_packet:
                payload = item.value if hasattr(item, "value") else item
                print(f"\n{payload.get('question')}")
                print(f"\n--- Draft Report ---\n{payload.get('draft_report')}\n--------------------")
            
            choice = input("\nOptions -> Type 'accept' or provide revision feedback: ").strip()
            
            if choice.lower() == "accept":
                # Resuming with accept tells the graph to finish (go to END)
                current_input = Command(resume={"status": "accept"})
            else:
                feedback_text = choice or "Please revise using corpus documents."
                # Resuming with revise tells the graph to loop back to the researcher
                current_input = Command(resume={"status": "revise", "feedback": feedback_text})
                print("\n🔄 Sending revision feedback back to Researcher...")

if __name__ == "__main__":
    run_task5_session()