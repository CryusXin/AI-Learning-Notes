import os

from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI

load_dotenv()
azure_chat = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)


def test1():
    # azure_ai 模型封装
    response = azure_chat.invoke("你是谁?")
    print(response)


def test2():
    # 多轮对话 session封装
    from langchain.schema import (
        AIMessage,  # 等价于OpenAI接口中的assistant role
        HumanMessage,  # 等价于OpenAI接口中的user role
        SystemMessage  # 等价于OpenAI接口中的system role
    )

    messages = [
        SystemMessage(content="你是ISTON的英语考官。"),
        HumanMessage(content="我是学生，我叫王四。"),
        AIMessage(content="欢迎！"),
        HumanMessage(content="我是谁")
    ]
    response = azure_chat.invoke(messages)
    print(response)


def test3():
    # prompt 模版封装
    from langchain.prompts import PromptTemplate

    template = PromptTemplate.from_template("给我讲个关于{subject}的{event}")
    print(template)
    print(template.format(subject='小明', event='故事'))
    print('-' * 20)
    from langchain.prompts import ChatPromptTemplate
    from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )

    prompt = template.format_messages(
        product="iston",
        name="谭桑",
        query="你是谁"
    )

    response = azure_chat.invoke(prompt)
    print(response)

if __name__ == '__main__':
    # test1()
    # test2()
    test3()
