from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
import yaml

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
