from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import os


set_tracing_disabled(True)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model_name = os.getenv("GEMINI_MODEL_NAME")


client = AsyncOpenAI(api_key=api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model=model_name
)

agent = Agent(
    name="FAQ Bot",
    instructions="You are a helpful FAQ bot Only based on the predefined questions.",
    model=model
)

questions = {
    "What is your name?": "I am a helpful agent",
    "What can you do?": "I can answer your questions clearly and helpfully.",
    "How do I reset my password?": "Click on Forgot Password at the login page and follow the instructions.",
    "Do you support multiple languages?": "Yes, I support multiple languages to assist users from different regions.",
    "Who created you?": "I was created by developers using the OpenAI Agent SDK to help answer common questions.",
}

print("FAQ Bot Responses:\n" + "-"*40)
for q in questions:
    result = Runner.run_sync(starting_agent=agent, input=q)
    print(f"\nYou: {q}\nBot: {result.final_output}")
