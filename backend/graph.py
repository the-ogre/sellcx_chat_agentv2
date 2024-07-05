import os
import operator
import logging
import uuid
from typing import Annotated, Sequence, TypedDict, Literal, List, Optional

from agent_chat import RAGAgent, Settings
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model with streaming enabled
try:
    model = ChatOpenAI(
        temperature=0.0,
        model_name='gpt-4o',
        openai_api_key=OPENAI_API_KEY
    )
    logger.info("ChatOpenAI model initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize ChatOpenAI model: %s", str(e))
    raise

# Define custom types as Pydantic models
class Image(BaseModel):
    url: str = Field(description="URL of the image")
    alt_text: Optional[str] = Field(description="Alternative text for the image", default=None)

class Table(BaseModel):
    headers: List[str] = Field(description="Headers of the table")
    rows: List[List[str]] = Field(description="Rows of the table")

class UIElement(BaseModel):
    type: str = Field(description="Type of the UI element")
    properties: dict = Field(description="Properties of the UI element")

# Define the response schema
class Response(BaseModel):
    text: str = Field(description="Text to include in the response")
    images: List[Image] = Field(default_factory=list, description="List of images to include in the response")
    tables: List[Table] = Field(default_factory=list, description="List of tables to include in the response")
    ui_elements: List[UIElement] = Field(default_factory=list, description="List of other UI elements to include in the response")

# Bind the response schema to the model
try:
    model = model.bind_tools([Response])
    logger.info("Response schema bound to model successfully")
except Exception as e:
    logger.exception("Failed to bind response schema to model: %s", str(e))
    raise

# Define AgentState TypedDict
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    chat_history: Optional[List[BaseMessage]]

# Function to determine whether to continue or end the workflow
def should_continue(state) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the last message is an AIMessage, it may have tool calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "Response":
            return "end"
        return "continue"
    return "end"

def call_model(state):
    """
    Calls the model with the current state messages and chat history.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary with the model's response messages.
    """
    try:
        settings = Settings()  # Initialize settings from environment variables
        rag_agent = RAGAgent(settings=settings)
        response = rag_agent.handle_message(state)
        return response
    except Exception as e:
        logger.exception("Error calling model: %s", str(e))
        raise

def call_tool(state):
    """
    Executes tools based on the current state messages.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary with the tool execution messages.
    """
    try:
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
            tool_invocations = [
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"],
                )
                for tool_call in last_message.tool_calls
            ]
            tool_executor = ToolExecutor([Response])  # Ensure tool_executor is defined and includes the Response tool
            responses = tool_executor.batch(tool_invocations, return_exceptions=True)
            tool_messages = [
                ToolMessage(
                    content=str(response),
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
                for tc, response in zip(last_message.tool_calls, responses)
            ]
            # Update chat history with the tool messages
            state["chat_history"].extend(tool_messages)
            return {"messages": tool_messages}
        return {"messages": []}
    except Exception as e:
        logger.exception("Error executing tools: %s", str(e))
        raise

def create_workflow():
    """
    Creates and compiles the state graph workflow.

    Returns:
        The compiled workflow application.
    """
    workflow = StateGraph(AgentState)

    # Add nodes to the workflow
    workflow.add_node("rag_agent", call_model)
    workflow.add_node("action", call_tool)

    # Set the entry point
    workflow.set_entry_point("rag_agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "rag_agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Add normal edge from action to rag_agent
    workflow.add_edge("action", "rag_agent")

    # Compile the workflow
    try:
        app = workflow.compile()
        logger.info("Workflow compiled successfully")
        return app
    except Exception as e:
        logger.exception("Failed to compile workflow: %s", str(e))
        raise


# import os
# import operator
# import logging
# import uuid
# from typing import Annotated, Sequence, TypedDict, Literal, List, Optional

# from agent_chat import RAGAgent, Settings
# from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt import ToolInvocation, ToolExecutor
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize the environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize the ChatOpenAI model with streaming enabled
# try:
#     model = ChatOpenAI(
#         temperature=0.0,
#         model_name='gpt-4o',
#         openai_api_key=OPENAI_API_KEY
#     )
#     logger.info("ChatOpenAI model initialized successfully")
# except Exception as e:
#     logger.exception("Failed to initialize ChatOpenAI model: %s", str(e))
#     raise

# # Define custom types as Pydantic models
# class Image(BaseModel):
#     url: str = Field(description="URL of the image")
#     alt_text: Optional[str] = Field(description="Alternative text for the image", default=None)

# class Table(BaseModel):
#     headers: List[str] = Field(description="Headers of the table")
#     rows: List[List[str]] = Field(description="Rows of the table")

# class UIElement(BaseModel):
#     type: str = Field(description="Type of the UI element")
#     properties: dict = Field(description="Properties of the UI element")

# # Define the response schema
# class Response(BaseModel):
#     text: str = Field(description="Text to include in the response")
#     images: List[Image] = Field(default_factory=list, description="List of images to include in the response")
#     tables: List[Table] = Field(default_factory=list, description="List of tables to include in the response")
#     ui_elements: List[UIElement] = Field(default_factory=list, description="List of other UI elements to include in the response")

# # Bind the response schema to the model
# try:
#     model = model.bind_tools([Response])
#     logger.info("Response schema bound to model successfully")
# except Exception as e:
#     logger.exception("Failed to bind response schema to model: %s", str(e))
#     raise

# # Define AgentState TypedDict
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]

# # Function to determine whether to continue or end the workflow
# def should_continue(state) -> Literal["continue", "end"]:
#     messages = state["messages"]
#     last_message = messages[-1]
#     # If the last message is an AIMessage, it may have tool calls
#     if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         if last_message.tool_calls[0]["name"] == "Response":
#             return "end"
#         return "continue"
#     return "end"

# def call_model(state):
#     """
#     Calls the model with the current state messages.

#     Args:
#         state: The current state of the workflow.

#     Returns:
#         A dictionary with the model's response messages.
#     """
#     try:
#         settings = Settings()  # Initialize settings from environment variables
#         rag_agent = RAGAgent(settings=settings)
#         response = rag_agent.handle_message(state)
#         return response
#     except Exception as e:
#         logger.exception("Error calling model: %s", str(e))
#         raise

# def call_tool(state):
#     """
#     Executes tools based on the current state messages.

#     Args:
#         state: The current state of the workflow.

#     Returns:
#         A dictionary with the tool execution messages.
#     """
#     try:
#         messages = state["messages"]
#         last_message = messages[-1]
#         if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
#             tool_invocations = [
#                 ToolInvocation(
#                     tool=tool_call["name"],
#                     tool_input=tool_call["args"],
#                 )
#                 for tool_call in last_message.tool_calls
#             ]
#             tool_executor = ToolExecutor([Response])  # Ensure tool_executor is defined and includes the Response tool
#             responses = tool_executor.batch(tool_invocations, return_exceptions=True)
#             tool_messages = [
#                 ToolMessage(
#                     content=str(response),
#                     name=tc["name"],
#                     tool_call_id=tc["id"],
#                 )
#                 for tc, response in zip(last_message.tool_calls, responses)
#             ]
#             return {"messages": tool_messages}
#         return {"messages": []}
#     except Exception as e:
#         logger.exception("Error executing tools: %s", str(e))
#         raise

# def create_workflow():
#     """
#     Creates and compiles the state graph workflow.

#     Returns:
#         The compiled workflow application.
#     """
#     workflow = StateGraph(AgentState)

#     # Add nodes to the workflow
#     workflow.add_node("rag_agent", call_model)
#     workflow.add_node("action", call_tool)

#     # Set the entry point
#     workflow.set_entry_point("rag_agent")

#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "rag_agent",
#         should_continue,
#         {
#             "continue": "action",
#             "end": END,
#         },
#     )

#     # Add normal edge from action to rag_agent
#     workflow.add_edge("action", "rag_agent")

#     # Compile the workflow
#     try:
#         app = workflow.compile()
#         logger.info("Workflow compiled successfully")
#         return app
#     except Exception as e:
#         logger.exception("Failed to compile workflow: %s", str(e))
#         raise
