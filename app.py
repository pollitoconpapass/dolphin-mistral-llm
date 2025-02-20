import chainlit as cl
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict


model = OllamaLLM(model="dolphin-mistral",)

@cl.on_chat_start
async def on_chat_start():
    history = ChatMessageHistory()
    cl.user_session.set("message_history", history)

    msg = cl.Message(content="Hello 👋, how can I help you? 🐬🇫🇷")
    await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.add_user_message(message.content)

    messages = messages_to_dict(history.messages)

    if len(messages) > 5:
        messages = messages[-5:]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You're an expert AI Assistant called DolphinMistral. You can answer any question the user asks you."""
            )
        ] + messages_from_dict(messages)
    )

    runnable = prompt | model | StrOutputParser()
    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    history.add_ai_message(msg.content)
    cl.user_session.set("message_history", history)
