import dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import httpx

# Load .env (where your OPENAI_API_KEY should live)
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")  # fetches your key as a string

print(api_key)

# Optional: disable SSL verify for dev (⚠️ not for prod)
client = httpx.Client(verify=False)

# Define your chat model using custom httpx client
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client  # << key addition
)

# Define system and human messages
system_message_news = """
You are a conversational AI assistant that recommends news articles based on the user's emotional state, interests, and current context (like sleep, energy, and mood).
Your job is to:
1. Understand how the user is feeling and what kind of news they are open to.
2. Recommend 1–3 articles from your internal database or memory that match the user's current state.
3. Provide a very short, human-friendly summary of each article in 1–2 sentences.
4. Explain why you're recommending that article — show empathy and relevance.
5. Keep the tone warm, casual, and thoughtful — like a friend who knows good stories.

Guidelines:
- Avoid harsh or triggering topics if the user feels sad, tired, or anxious.
- If the user is energized or curious, you can suggest deeper, more challenging content.
- Prioritize diversity: recommend a mix of topics (science, human stories, tech, culture) when possible.
- Speak naturally. Be kind. You're here to help the user feel more connected and informed, not overwhelmed
"""

human_message_news = "I barely slept last night and I just feel kind of overwhelmed. I want to stay informed but not get more stressed. Got anything chill?"

messages = [
    SystemMessage(content=system_message_news),
    HumanMessage(content=human_message_news)
]

# Invoke the chat
response = chat_model.invoke(messages)
print(response.content)
