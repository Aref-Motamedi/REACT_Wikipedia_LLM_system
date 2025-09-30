import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint
from langchain.tools import Tool

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # You can change this model
    temperature=0.1,  # Lower = more focused, Higher = more creative
    max_new_tokens=512,
)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
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

