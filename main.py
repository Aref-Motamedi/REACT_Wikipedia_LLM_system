import os
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import Ollama
from langchain.tools import Tool

load_dotenv()

llm = Ollama(
    model="mistral",
    temperature = 0.3,
    # stop=["Observation:", "\nObservation"]  # Stop after seeing these

)

# llm = HuggingFaceEndpoint(
#     # repo_id="google/flan-t5-xxl",  
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     temperature=0.1,  # Lower = more focused, Higher = more creative
#     max_new_tokens=512,
# )

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=500,
    )
)

def calculator(expression):
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating {str(e)}"

calculator_tool = Tool(
    name='Calculator',
    func=calculator,
    description='Does mathematical calculation (valid python mathematical expressions)'
)

tools = [wikipedia,calculator_tool]

react_prompt = PromptTemplate.from_template("""
Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: After you receive an Observation, you MUST either:
1. Use another tool if you need more information, OR
2. Provide the Final Answer if you have enough information

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


agent = create_react_agent(
    llm = llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",  # Forces final answer
    return_intermediate_steps=True  # See what went wrong
)

# test_queries = [
#     "What is the population of Tokyo? Multiply it by 2.",
#     "Who was Albert Einstein and when was he born?",
#     "Calculate 157 * 89 and tell me if it's greater than 10000"
# ]


# for query in test_queries:
#     print(f"\n{'='*50}")
#     print(f"QUERY: {query}")
#     print('='*50)
    
#     try:
#         response = agent_executor.invoke({"input": query})
#         print(f"\nFINAL ANSWER: {response['output']}")
#     except Exception as e:
#         print(f"Error: {str(e)}")
    
#     print()


print("\n" + "="*50)
print("Interactive Mode - Type 'quit' to exit")
print("="*50 + "\n")

while True:
    user_input = input("\nYour question: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"\n✅ Answer: {response['output']}\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
