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


#%%
# ==========================================
# 2. Define AgentState (Shared State Schema)
# ==========================================
# AgentState is a TypedDict that holds our message history.
# The annotated lambda function (x, y: x + y) acts as a "reducer",
# meaning new messages are appended to the existing list rather than overwriting it.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]


# ==========================================
# 3. Define Graph Nodes (Tasks / Actions)
# ==========================================

# Node A: The Agent/Planning Node
# It invokes the LLM with system instructions and chat history to decide what to do next.
def call_model(state: AgentState):


    messages = state["messages"]
    
    system_instruction = (
        "You are an advanced research assistant. You MUST use the available tools to answer questions.\n"
        "1. **MANDATORY**: For ANY questions regarding internal documents, local project files, company policies (like PTO, HR rules), or knowledge base, you MUST use `document_search` first.\n"
        "2. Use `web_search` ONLY for live public internet data, current events, or general knowledge.\n"
        "3. Use `calculate` for math expressions."
    )
    
    full_messages = [{"role": "system", "content": system_instruction}] + list(messages)
    response = llm_with_tools.invoke(full_messages)
    
    # Returns the new AI message wrapped in a list to append to the state
    return {"messages": [response]}


# Node B: The Tool Execution Node
# LangGraph provides a built-in `ToolNode` that automatically runs whatever tool 
# the LLM requested in its last message.
tool_node = ToolNode(tools=tools)

#  4. Define Conditional Routing Logic
# ==========================================
# This function inspects the last message to decide where the graph should go next:
# - If the LLM called a tool -> go to the "tools" node.
# - If the LLM gave a final text answer -> stop execution (END).
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END


#  5. Build and Compile the Graph app
# ==========================================
app = StateGraph(AgentState)

# Step 5.1: Add our nodes to the graph
app.add_node("agent", call_model)
app.add_node("tools", tool_node)

# Step 5.2: Set entry point (where execution always starts)
app.add_edge(START, "agent")

# Step 5.3: Add conditional edges from the agent node
app.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Step 5.4: After tools execute, always loop back to the agent 
# so it can read the tool's observation and form a response.
app.add_edge("tools", "agent")

# Step 5.5: Add an in-memory checkpointer so state/history persists across turns
memory = MemorySaver()
app = app.compile(checkpointer=memory)
# app = app.compile()


#%%
# ==========================================
# 6. Interactive Session Runner Loop
# ==========================================
def run_task3_session():
    print("\n--- Task 3: LangGraph State Machine Agent ---")
    
    # Thread config enables state persistence across multiple questions in the same session
    thread_config = {"configurable": {"thread_id": "session-1"}}

    try:
        limit_input = input("How many questions would you like to ask in this session? (Enter a number): ").strip()
        max_questions = int(limit_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 questions.")
        max_questions = 5

    print(f"\nSession initialized! You can ask up to {max_questions} questions.")
    print("Type 'exit' or 'quit' to end early.\n")

    question_count = 0

    while question_count < max_questions:
        try:
            remaining = max_questions - question_count
            user_query = input(f"\n[Question {question_count + 1} of {max_questions}] (Remaining: {remaining}) Ask a question: ").strip()
            
            if user_query.lower() in ["exit", "quit"]:
                print("\nExiting session. Goodbye!")
                break
                
            if not user_query:
                print("Please enter a valid question.")
                continue
                
            print(f"\nProcessing through LangGraph State Machine...")
            
            input_message = HumanMessage(content=user_query)
            final_output = None
            
            # Stream the graph execution step-by-step
            for event in app.stream({"messages": [input_message]}, thread_config, stream_mode="values"):
                latest_message = event["messages"][-1]
                if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
                    print(f"  ⚡ Action: Invoking tool -> {latest_message.tool_calls}")
                elif isinstance(latest_message, ToolMessage):
                    print(f"  📦 Observation received from tool.")
                final_output = latest_message.content

            print(f"\nAnswer: {final_output}\n")
            print("=" * 60)
            
            question_count += 1
            
        except KeyboardInterrupt:
            print("\nExiting session. Goodbye!")
            break
        except Exception as e:
            print(f"\nExecution Error: {e}\n" + "=" * 60)
            question_count += 1

    if question_count >= max_questions:
        print(f"\n--- Reached limit of {max_questions} questions! Task 3 session closed. ---")

if __name__ == "__main__":
    run_task3_session()