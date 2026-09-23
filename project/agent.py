import os
import sys
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure 'project' is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) # project/
root_dir = os.path.dirname(current_dir) # Multi_agent_researcher/
sys.path.insert(0, root_dir)

from langchain_classic.agents import create_react_agent, AgentExecutor,create_tool_calling_agent
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from project.tools.calculator import calculate
from project.config.setting import config,get_llm
from project.tools.search import web_search_tool
from project.tools.document_search import document_search
from project.services.vector_store import get_retriever

def task1():
    print("Checking and initializing vector store and embedding models...")
    # This will download the embedder to local_models if not already downloaded,
    # and build the vector store if it's empty.
    get_retriever()
    
    # 1. Initialize LLM via settings.py
    llm = get_llm()

    # 2. Gather Tools (Web Search, Calculator, Document Search)
    tools = [web_search_tool, calculate, document_search]

    # 3. Define Standard ReAct Prompt Template
    # FIX: The original prompt used "TOOL:" which the AgentExecutor cannot parse.
    # The ReAct parser strictly expects "Action:" and "Action Input:" keywords.
    # Extra blank lines between steps also break the parser.
    # 3. Define Standard ReAct Prompt Template
   
    # 3. Use a Modern Chat Prompt Template for Tool Calling Agents
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced research assistant with strict tool-routing guidelines:\n"
                   "1. **MANDATORY**: For ANY questions regarding policies (such as PTO, HR, leave, benefits), company documents, project files, or local knowledge, you MUST use `document_search` first.\n"
                   "2. Do NOT use `web_search` for internal policies or company handbook topics.\n"
                   "3. Use `web_search` ONLY for live public internet data, current news, or general public knowledge.\n"
                   "4. Use `calculate` for math expressions."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Construct Modern Tool-Calling Agent (No more parsing crashes!)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True
    )




#     prompt = PromptTemplate.from_template(
#         """You are an advanced research assistant. You MUST use the available tools to answer questions. Never answer from memory alone.

# TOOLS:
# ------
# You have access to the following tools:

# {tools}

# To use a tool, please use the following strict format:

# Question: the input question you must answer
# Thought: think about which tool to use and why
# Action: the tool name, must be one of [{tool_names}]
# Action Input: the input string to pass to the tool
# Observation: the result returned by the tool
# ... (repeat Thought/Action/Action Input/Observation as needed)
# Thought: I now know the final answer
# Final Answer: [Tools used: <comma-separated tool names>] <your complete answer here. If document_search was used, mention the source filename.>

# TOOL ROUTING RULES (CRITICAL):
# 1. **document_search**: MUST be used first for any questions regarding internal documents, project code, uploaded files, or local knowledge base.
# 2. **web_search**: Use ONLY for live public internet searches, current events, or general knowledge not found locally.
# 3. **calculate**: Use ONLY for evaluating mathematical expressions.

# CRITICAL FORMATTING CONSTRAINTS:
# - ALWAYS use a tool at least once before writing Final Answer.
# - Action must be EXACTLY one of: [{tool_names}].
# - Output ONLY ONE step at a time. After Action + Action Input, STOP and wait for the Observation.
# - Your Final Answer must start with `[Tools used: ...]`.

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""
#     )

#     # 4. Construct ReAct Agent
#     agent = create_react_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=True,
#         # FIX: Raised from 5 to 12 — complex queries need more reasoning steps before finishing
#         max_iterations=12,
#         handle_parsing_errors=True
#     )

    # 5. Ask user for the maximum number of questions limit
    print("\n--- Task 1: LangChain ReAct Tool-Calling Agent ---")
    try:
        limit_input = input("How many questions would you like to ask in this session? (Enter a number): ").strip()
        max_questions = int(limit_input)
    except ValueError:
        print("Invalid number entered. Defaulting to a limit of 5 questions.")
        max_questions = 5

    print(f"\nSession initialized! You can ask up to {max_questions} questions.")
    print("Type 'exit' or 'quit' at any time to end early.\n")

    question_count = 0

    # 6. Controlled Dynamic Question Loop
    while question_count < max_questions:
        try:
            remaining = max_questions - question_count
            user_query = input(f"\n[Question {question_count + 1} of {max_questions}] (Remaining: {remaining}) Ask a question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ["exit", "quit"]:
                print("\n Exiting agent session early. Goodbye!")
                break
                
            if not user_query:
                print("Please enter a valid question.")
                continue
                
            print(f"\nProcessing: {user_query}")
            response = agent_executor.invoke({"input": user_query})
            print(f"\nAnswer: {response['output']}\n\n")
            print("=" * 60)
            
            # Increment successful question counter
            question_count += 1
            
        except KeyboardInterrupt:
            print("\nExiting agent session. Goodbye!")
            break
        except Exception as e:
            print(f"\nExecution Error: {e}\n" + "=" * 60)
            question_count += 1

    if question_count >= max_questions:
        print(f"\n--- Reached your limit of {max_questions} questions! Task 1 session closed. ---")

if __name__ == "__main__":
    task1()