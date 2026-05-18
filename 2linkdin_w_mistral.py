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

# LinkedIn Post Prompt Template
prompt = PromptTemplate(
    input_variables=["topic", "tone", "target"],
    template="""
You are an expert LinkedIn Content Strategist. Your goal is to write a highly engaging, viral-worthy LinkedIn post.

Topic: {topic}
Tone: {tone}
Target Audience: {target}

Guidelines:
1. Start with a strong hook to stop the scroll.
2. Use short, punchy sentences for readability.
3. Use bullet points if applicable.
4. Add relevant hashtags (max 3-5).
5. Include a clear Call to Action (CTA) at the end.
6. Ensure the tone matches the requested style.

Write the post below:
"""
)

# User Inputs
topic = input("Enter the Topic for the LinkedIn post: ")
tone = input("Enter the Tone (e.g., Professional, Conversational, Witty, Inspirational): ")
target = input("Enter the Target Audience (e.g., Tech Founders, HR Professionals, Students): ")

# Create Final Prompt
final_prompt = prompt.format(
    topic=topic,
    tone=tone,
    target=target
)

# Generate Response
print("\nGenerating your LinkedIn post... \n")
response = llm.invoke(final_prompt)

# Print Output
print("Generated LinkedIn Post:")
print(response.content)