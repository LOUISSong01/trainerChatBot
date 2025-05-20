import dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import httpx

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
#from langchain_intro.tools import get_current_wait_time

# Load .env (where your OPENAI_API_KEY should live)
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # fetches your key as a string

# # Optional: disable SSL verify for dev (⚠️ not for prod)
# client = httpx.Client(verify=False)

# Define your chat model using custom httpx client
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Define system and human messages
# system_message_news = """
# You are a conversational AI assistant that recommends news articles based on the user's emotional state, interests, and current context (like sleep, energy, and mood).
# Your job is to: 
# """

# human_message_news = "I barely slept last night and I just feel kind of overwhelmed. I want to stay informed but not get more stressed. Got anything chill?"

# messages = [
#     SystemMessage(content=system_message_news),
#     HumanMessage(content=human_message_news)
# ]

#might add few-shot prompting here
news_system_template_str = """
You are a conversational AI assistant that recommends news articles based on the user's emotional state, interests, and current context (like sleep, energy, and mood). Very Important: You MUST NOT use any knowledge outside of the given context.
Only use the articles and facts provided in the context. Do not guess or assume anything.

If the context does not contain relevant information, respond with:
"I don't know" or "I can't help with that right now."

Even if the question seems easy, DO NOT generate an answer from your own knowledge. Stick strictly to the context."

Context: {context}

Follow these steps below to generate a response:
Guidelines: {steps}

Keep in mind to avoid doing these things:
{caution}
"""
#news_template = ChatPromptTemplate.from_template(new_template_str)

news_system_prompt = SystemMessagePromptTemplate(
    prompt = PromptTemplate(
        input_variables=["context", "steps", "caution"], template = news_system_template_str
    )
)

news_human_prompt = HumanMessagePromptTemplate(
    prompt = PromptTemplate(
        input_variables=["question"], template = "{question}"
    )
)

messages = [news_system_prompt, news_human_prompt]

news_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

# parameters list
context = "Recommendations for news articles based on the user's emotional state, interests, and current context (like sleep, energy, and mood)."

steps= """
1. Understand how the user is feeling and what kind of news they are open to.
2. Recommend 1–3 articles from your internal database or memory that match the user's current state.
3. Provide a very short, human-friendly summary of each article in 1–2 sentences.
4. Explain why you're recommending that article — show empathy and relevance.
5. Keep the tone warm, casual, and thoughtful — like a friend who knows good stories.
"""
question = "Hello!"

caution = """
- Avoid harsh or triggering topics if the user feels sad, tired, or anxious.
- If the user is energized or curious, you can suggest deeper, more challenging content.
- Prioritize diversity: recommend a mix of topics (science, human stories, tech, culture) when possible.
- Speak naturally. Be kind. You're here to help the user feel more connected and informed, not overwhelmed
"""

news_vector_db = Chroma (embedding_function=OpenAIEmbeddings(), persist_directory="chroma_data")
news_retriever = news_vector_db.as_retriever(search_kwargs={'k': 1})


chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature = 0)
output_parser = StrOutputParser()
# Inject fixed values
inject_constants = RunnableLambda(lambda x: {
    "question": x["question"],
    "context": x["context"],  # <- pass question string to retriever
    "steps": steps,
    "caution": caution
})


# Define chain
news_chain = {
    "context": RunnableLambda(lambda x: context) | news_retriever,
    "question": RunnablePassthrough()
} | inject_constants | news_prompt_template | chat_model | output_parser

#for agents use 
tools = [
    Tool(
        name="Articles",
        func=news_chain.invoke,
        description="A tool to recommend news articles based on the user's emotional state, interests, and current context (like sleep, energy, and mood).",
    )
]

agent_chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

article_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    
article_agent = create_openai_functions_agent(
    llm=agent_chat_model,
    tools=tools,
    prompt = article_agent_prompt,
)

article_agent_executor = AgentExecutor(
    agent=article_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

#print(news_chain.invoke({"question": question, "context": context}))
print(article_agent_executor.invoke({"input" : "Do you know any news article about apple bugs?"}))