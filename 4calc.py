from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import Tool
import os

load_dotenv()

tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),  # executes the math expression, e.g. "20 * 40"
        description="Useful for mathematical calculations. Input: a valid Python expression like '23 * 45'.",
    )
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm_with_tools = llm.bind_tools(tools)

question = input("Enter your question: ")

response = llm_with_tools.invoke(question)

if response.tool_calls:
    tool_call = response.tool_calls[0]  

    if tool_call["name"] == "Calculator":
        
        expr = tool_call["args"]["__arg1"]  
        result = tools[0].func(expr)       
        print("Result:", result)
    else:
       
        print("Unknown tool:", tool_call["name"])
else:
    
    print(response.content)