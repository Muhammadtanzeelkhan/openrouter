from dotenv import load_dotenv
import os

# Import classes from your `agents` package
from agents import Agent, Runner, RunConfig
from agents.openai_client import AsyncOpenAI  # Ensure 'openai_client.py' exists in the 'agents' package, or correct the import path if the file/module name is different
from agents.models import OpenAIChatCompletionsModel

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENAIROUTER_API_KEY")

# Check if the API key is set
if not openrouter_api_key:
    raise ValueError(
        "OPENAIROUTER_API_KEY is not set. Please ensure it is defined in your .env file."
    )

# External client for OpenRouter-compatible models
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Define the model configuration
model = OpenAIChatCompletionsModel(
    model="tngtech/deepseek-r1t-chimera:free",
    openai_client=external_client,
)

# Setup config
config = RunConfig(
    model=model,
    model_provider=model,
    tracing_disabled=True,
)

# Define the agent
agent = Agent(
    name="writer agent",
    instructions="You are a writer agent. Generate stories, poems, etc."
)

# Run the agent synchronously with input and config
response = Runner.run_sync(
    agent,
    input="Write a short thrilling fantastic movie script in advanced English.",
    run_config=config,
)

# Print the response
print(response)
