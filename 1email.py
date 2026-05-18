from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Initialize Mistral LLM
llm = ChatMistralAI(
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral-small-2506",
    temperature=0.7,
    max_tokens=500
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["content", "customer_name", "agent_name"],
    template="""
You are an AI assistant optimized to write concise, professional, and friendly emails.

Write a professional email to {customer_name}.

Content:
{content}

Signature:
{agent_name}
"""
)

# User Input
content = input("Email content: ")
customer = input("Customer name: ")
agent = input("Agent name: ")

# Create Final Prompt
final_prompt = prompt.format(
    content=content,
    customer_name=customer,
    agent_name=agent
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated Email:\n")
print(response.content)
