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
    input_variables=["situation", "bowler_name", "batter_name"],
    template="""
You are an exciting cricket commentator.

Create energetic and realistic cricket commentary based on the match situation.

Match Situation:
{situation}

Bowler Name:
{bowler_name}

Batter Name:
{batter_name}

The commentary should be:
- Exciting
- Natural
- Short and engaging
- Like live TV commentary
"""
)

# User Input
situation = input("Enter Match Situation: ")
bowler = input("Enter Bowler Name: ")
batter = input("Enter Batter Name: ")

# Create Final Prompt
final_prompt = prompt.format(
    situation=situation,
    bowler_name=bowler,
    batter_name=batter
)

# Print Prompt
print("\nFinal Prompt:\n")
print(final_prompt)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nCricket Commentary:\n")
print(response.content)
