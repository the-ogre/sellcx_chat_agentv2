import os
import uuid
import logging
from typing import Dict, Any
from neo4j import GraphDatabase
from pydantic import BaseModel, ValidationError, SecretStr
from pydantic_settings import BaseSettings
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from tool_retriever import retriever
from prompts import answer_template, question_template
from utilities import format_chat_history, load_yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    LANGCHAIN_TRACING_V2: bool
    LANGCHAIN_API_KEY: SecretStr
    LANGCHAIN_PROJECT: str
    OPENAI_API_KEY: SecretStr
    OPENAI_EMBEDDING_MODEL: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: SecretStr

    class Config:
        env_file = ".env"

settings = Settings()

class RAGAgent:
    def __init__(self, settings: Settings):
        """
        Initializes the RAGAgent, setting up the LLM chain and environment.
        """
        self.settings = settings
        self.chain = self.create_rag_chain()

    def create_rag_chain(self):
        """
        Creates the RAG chain for processing questions.

        Returns:
            The RAG chain object.
        """
        llm = ChatOpenAI(
            temperature=0.0,
            model_name='gpt-4o',
            openai_api_key=self.settings.OPENAI_API_KEY.get_secret_value()
        )

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_template)

        _search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | llm
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        answer_prompt = ChatPromptTemplate.from_template(answer_template)

        chain = (
            RunnableParallel(
                {
                    "context": _search_query | retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | answer_prompt
            | llm
            | StrOutputParser()
        )

        return chain

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles the incoming message and returns the response.

        Args:
            state: The current state containing messages and chat history.

        Returns:
            A dictionary with the tool's response message.
        """
        messages = state["messages"]
        chat_history = state.get("chat_history", [])
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage):
            question = last_message.content  # Extract the question from the HumanMessage content
        else:
            logger.error("Last message is not a HumanMessage")
            question = "No valid question found."

        try:
            # Invoke the RAG chain with the question and chat history
            result = self.chain.invoke({"question": question, "chat_history": chat_history})

            # Ensure result is a dictionary and extract the response content
            if isinstance(result, dict) and "output" in result:
                response_content = result["output"]
            else:
                response_content = str(result)  # Convert result to string if it's not a dictionary

            # Return the response as a ToolMessage with a unique tool_call_id
            return {
                "messages": [ToolMessage(content=response_content, name="ragagent", tool_call_id=str(uuid.uuid4()))]
            }
        except Exception as e:
            logger.exception("Error occurred while processing the RAG tool")
            return {
                "messages": [ToolMessage(content=f"Error: {str(e)}", name="ragagent", tool_call_id=str(uuid.uuid4()))]
            }

def main():
    try:
        settings = Settings()
    except ValidationError as e:
        logger.error("Configuration error: %s", e)
        raise

    agent = RAGAgent(settings=settings)
    
    # Dummy state for testing
    state = {
        "messages": [HumanMessage(content="Can you provide me a list of services you provide")],
        "chat_history": []
    }
    response = agent.handle_message(state)
    logger.info("Response: %s", response)

if __name__ == "__main__":
    main()


# import os
# import uuid
# import logging
# from typing import Dict, Any
# from neo4j import GraphDatabase
# from pydantic import BaseModel, ValidationError, SecretStr
# from pydantic_settings import BaseSettings
# from langchain_core.runnables import (
#     RunnableBranch,
#     RunnableLambda,
#     RunnableParallel,
#     RunnablePassthrough,
# )
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts.prompt import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, ToolMessage
# from tool_retriever import retriever
# from prompts import answer_template, question_template
# from utilities import format_chat_history, load_yaml
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class Settings(BaseSettings):
#     LANGCHAIN_TRACING_V2: bool
#     LANGCHAIN_API_KEY: SecretStr
#     LANGCHAIN_PROJECT: str
#     OPENAI_API_KEY: SecretStr
#     OPENAI_EMBEDDING_MODEL: str
#     NEO4J_URI: str
#     NEO4J_USERNAME: str
#     NEO4J_PASSWORD: SecretStr

#     class Config:
#         env_file = ".env"

# settings = Settings()

# class RAGAgent:
#     def __init__(self, settings: Settings):
#         """
#         Initializes the RAGAgent, setting up the LLM chain and environment.
#         """
#         self.settings = settings
#         self.chain = self.create_rag_chain()

#     def create_rag_chain(self):
#         """
#         Creates the RAG chain for processing questions.

#         Returns:
#             The RAG chain object.
#         """
#         llm = ChatOpenAI(
#             temperature=0.0,
#             model_name='gpt-4o',
#             openai_api_key=self.settings.OPENAI_API_KEY.get_secret_value()
#         )

#         CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_template)

#         _search_query = RunnableBranch(
#             (
#                 RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
#                     run_name="HasChatHistoryCheck"
#                 ),
#                 RunnablePassthrough.assign(
#                     chat_history=lambda x: format_chat_history(x["chat_history"])
#                 )
#                 | CONDENSE_QUESTION_PROMPT
#                 | llm
#                 | StrOutputParser(),
#             ),
#             RunnableLambda(lambda x: x["question"]),
#         )

#         answer_prompt = ChatPromptTemplate.from_template(answer_template)

#         chain = (
#             RunnableParallel(
#                 {
#                     "context": _search_query | retriever,
#                     "question": RunnablePassthrough(),
#                 }
#             )
#             | answer_prompt
#             | llm
#             | StrOutputParser()
#         )

#         return chain

#     def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Handles the incoming message and returns the response.

#         Args:
#             state: The current state containing messages.

#         Returns:
#             A dictionary with the tool's response message.
#         """
#         messages = state["messages"]
#         last_message = messages[-1]
        
#         if isinstance(last_message, HumanMessage):
#             question = last_message.content  # Extract the question from the HumanMessage content
#         else:
#             logger.error("Last message is not a HumanMessage")
#             question = "No valid question found."

#         try:
#             # Invoke the RAG chain with the question
#             result = self.chain.invoke({"question": question})

#             # Ensure result is a dictionary and extract the response content
#             if isinstance(result, dict) and "output" in result:
#                 response_content = result["output"]
#             else:
#                 response_content = str(result)  # Convert result to string if it's not a dictionary

#             # Return the response as a ToolMessage with a unique tool_call_id
#             return {
#                 "messages": [ToolMessage(content=response_content, name="ragagent", tool_call_id=str(uuid.uuid4()))]
#             }
#         except Exception as e:
#             logger.exception("Error occurred while processing the RAG tool")
#             return {
#                 "messages": [ToolMessage(content=f"Error: {str(e)}", name="ragagent", tool_call_id=str(uuid.uuid4()))]
#             }

# def main():
#     try:
#         settings = Settings()
#     except ValidationError as e:
#         logger.error("Configuration error: %s", e)
#         raise

#     agent = RAGAgent(settings=settings)
    
#     # Dummy state for testing
#     state = {
#         "messages": [HumanMessage(content="Can you provide me a list of services you provide")]
#     }
#     response = agent.handle_message(state)
#     logger.info("Response: %s", response)

# if __name__ == "__main__":
#     main()
