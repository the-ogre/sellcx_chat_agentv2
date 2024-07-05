import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from graph import create_workflow

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Interaction Bot v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}
SESSION_ID = None

class UserMessage(BaseModel):
    session_id: str = None
    message: str

@app.on_event("startup")
async def startup_event():
    global SESSION_ID
    # Ensure the sessions directory exists
    os.makedirs("SESSIONS", exist_ok=True)

    # Always create a new session with a unique ID at startup
    SESSION_ID = str(uuid.uuid4())
    sessions[SESSION_ID] = {"chat_history": [], "state": {}}
    save_new_session(SESSION_ID)

def save_new_session(session_id):
    try:
        session_data = {"session_id": session_id, "chat_history": sessions[session_id]["chat_history"], "state": sessions[session_id]["state"]}
        with open(f"SESSIONS/{session_id}.json", "w") as file:
            json.dump(session_data, file, indent=4)
    except Exception as e:
        print(f"Failed to save session {session_id}: {str(e)}")

@app.get("/")
async def read_root():
    welcome_message = '''
    Hi folks! We are Highland Constructors
    '''
    return {"message": welcome_message}

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "chat_history": session["chat_history"]}

@app.get("/state")
async def get_state():
    return {"states": {sid: session["state"] for sid, session in sessions.items()}}

@app.post("/send-message")
async def send_message(user_message: UserMessage):
    try:
        # Use the provided session ID or create a new one if not provided
        session_id = user_message.session_id or SESSION_ID
        if not session_id or session_id not in sessions:
            session_id = SESSION_ID
            sessions[session_id] = {"chat_history": [], "state": {}}

        user_content = user_message.message
        sessions[session_id]["chat_history"].append({"sender": "user", "content": user_content})

        # Create state for workflow
        state = {
            "messages": [
                HumanMessage(content=msg["content"]) if msg["sender"] == "user"
                else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
                for msg in sessions[session_id]["chat_history"]
            ]
        }

        # Run workflow
        workflow_app = create_workflow()
        response = workflow_app.invoke(state)

        # Ensure only new bot messages are appended to chat history
        new_messages = [
            {"sender": "bot", "content": message.content}
            for message in response["messages"]
            if not isinstance(message, HumanMessage)  # Only include non-human messages
        ]
        
        sessions[session_id]["chat_history"].extend(new_messages)
        sessions[session_id]["state"] = state
        
        last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

        return JSONResponse(content={"message": last_message["content"], "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# import uuid
# import json
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, ToolMessage
# from graph import create_workflow

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Interaction Bot v1")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory session storage
# sessions = {}

# class UserMessage(BaseModel):
#     session_id: str = None
#     message: str

# @app.on_event("startup")
# async def startup_event():
#     try:
#         with open("chat_history.json", "r") as file:
#             chat_history_data = json.load(file)
#             for session in chat_history_data["sessions"]:
#                 if not session.get("session_id"):
#                     session["session_id"] = str(uuid.uuid4())
#                 sessions[session["session_id"]] = {"chat_history": session["chat_history"], "state": {}}
#     except FileNotFoundError:
#         print("chat_history.json not found. Starting with empty sessions.")
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Failed to decode chat history file.")

# @app.get("/")
# async def read_root():
#     welcome_message = '''
#     Hi folks! We are Highland Constructors
#     '''
#     return {"message": welcome_message}

# @app.get("/chat-history")
# async def get_chat_history():
#     return {"sessions": [{"session_id": session_id, "chat_history": session["chat_history"]} for session_id, session in sessions.items()]}

# @app.get("/state")
# async def get_state():
#     return {"states": {sid: session["state"] for sid, session in sessions.items()}}

# @app.post("/send-message")
# async def send_message(user_message: UserMessage):
#     try:
#         # Use the provided session ID or create a new one if not provided
#         session_id = user_message.session_id
#         if not session_id or session_id not in sessions:
#             session_id = str(uuid.uuid4())
#             sessions[session_id] = {"chat_history": [], "state": {}}

#         user_content = user_message.message
#         sessions[session_id]["chat_history"].append({"sender": "user", "content": user_content})

#         # Create state for workflow
#         state = {
#             "messages": [
#                 HumanMessage(content=msg["content"]) if msg["sender"] == "user"
#                 else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
#                 for msg in sessions[session_id]["chat_history"]
#             ]
#         }

#         # Run workflow
#         workflow_app = create_workflow()
#         response = workflow_app.invoke(state)

#         # Ensure only new bot messages are appended to chat history
#         new_messages = [
#             {"sender": "bot", "content": message.content}
#             for message in response["messages"]
#             if not isinstance(message, HumanMessage)  # Only include non-human messages
#         ]
        
#         sessions[session_id]["chat_history"].extend(new_messages)
#         sessions[session_id]["state"] = state
        
#         last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

#         return JSONResponse(content={"message": last_message["content"], "session_id": session_id})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import os
# import uuid
# import json
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, ToolMessage
# from graph import create_workflow

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Interaction Bot v1")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory session storage
# sessions = {}

# class UserMessage(BaseModel):
#     message: str

# @app.on_event("startup")
# async def startup_event():
#     try:
#         with open("chat_history.json", "r") as file:
#             chat_history_data = json.load(file)
#             for session in chat_history_data["sessions"]:
#                 sessions[session["session_id"]] = {"chat_history": session["chat_history"], "state": {}}
#     except FileNotFoundError:
#         print("chat_history.json not found. Starting with empty sessions.")
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Failed to decode chat history file.")

# @app.get("/")
# async def read_root():
#     welcome_message = '''
#     Hi folks! We are Highland Constructors
#     '''
#     return {"message": welcome_message}

# @app.get("/chat-history")
# async def get_chat_history():
#     return {"sessions": [{"session_id": session_id, "chat_history": session["chat_history"]} for session_id, session in sessions.items()]}

# @app.get("/state")
# async def get_state():
#     return {"states": {sid: session["state"] for sid, session in sessions.items()}}

# @app.post("/send-message")
# async def send_message(user_message: UserMessage):
#     try:
#         # Generate a session ID if it doesn't exist
#         session_id = str(uuid.uuid4())

#         # Check if the session already exists
#         if session_id not in sessions:
#             sessions[session_id] = {"chat_history": [], "state": {}}

#         user_content = user_message.message
#         sessions[session_id]["chat_history"].append({"sender": "user", "content": user_content})

#         # Create state for workflow
#         state = {
#             "messages": [
#                 HumanMessage(content=msg["content"]) if msg["sender"] == "user"
#                 else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
#                 for msg in sessions[session_id]["chat_history"]
#             ]
#         }

#         # Run workflow
#         workflow_app = create_workflow()
#         response = workflow_app.invoke(state)

#         # Ensure only new bot messages are appended to chat history
#         new_messages = [
#             {"sender": "bot", "content": message.content}
#             for message in response["messages"]
#             if not isinstance(message, HumanMessage)  # Only include non-human messages
#         ]
        
#         sessions[session_id]["chat_history"].extend(new_messages)
#         sessions[session_id]["state"] = state
        
#         last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

#         return JSONResponse(content={"message": last_message["content"]})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# import os
# import uuid
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from pydantic_settings import BaseSettings
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, ToolMessage
# from graph import create_workflow

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Interaction Bot v1")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory session storage
# sessions = {}

# class UserMessage(BaseModel):
#     message: str

# @app.get("/")
# async def read_root():
#     welcome_message = '''
#     Hi folks! We are Highland Constructors
#     '''
#     return {"message": welcome_message}

# @app.get("/chat-history")
# async def get_chat_history():
#     return {"sessions": [{"chat_history": session["chat_history"]} for session in sessions.values()]}

# @app.get("/state")
# async def get_state():
#     return {"states": {sid: session["state"] for sid, session in sessions.items()}}

# @app.post("/send-message")
# async def send_message(user_message: UserMessage):
#     try:
#         # Generate a session ID if it doesn't exist
#         session_id = str(uuid.uuid4())
#         if session_id not in sessions:
#             sessions[session_id] = {"chat_history": [], "state": {}}

#         user_content = user_message.message
#         sessions[session_id]["chat_history"].append({"sender": "user", "content": user_content})

#         # Create state for workflow
#         state = {
#             "messages": [
#                 HumanMessage(content=msg["content"]) if msg["sender"] == "user"
#                 else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
#                 for msg in sessions[session_id]["chat_history"]
#             ]
#         }

#         # Run workflow
#         workflow_app = create_workflow()
#         response = workflow_app.invoke(state)

#         # Ensure only new bot messages are appended to chat history
#         new_messages = [
#             {"sender": "bot", "content": message.content}
#             for message in response["messages"]
#             if not isinstance(message, HumanMessage)  # Only include non-human messages
#         ]
        
#         sessions[session_id]["chat_history"].extend(new_messages)
#         sessions[session_id]["state"] = state
        
#         last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

#         return JSONResponse(content={"message": last_message["content"]})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# import uuid
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from pydantic_settings import BaseSettings
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, ToolMessage
# from graph import create_workflow

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Interaction Bot v1")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory session storage
# sessions = {}

# class UserMessage(BaseModel):
#     message: str

# @app.get("/")
# async def read_root():
#     welcome_message = '''
#     Hi folks! We are Highland Constructors
#     '''
#     return {"message": welcome_message}

# @app.get("/chat-history")
# async def get_chat_history():
#     return sessions

# @app.get("/state/{session_id}")
# async def get_state(session_id: str):
#     if session_id in sessions:
#         return sessions[session_id]
#     else:
#         raise HTTPException(status_code=404, detail="Session not found")

# @app.post("/send-message")
# async def send_message(user_message: UserMessage):
#     try:
#         # Generate a session ID if it doesn't exist
#         session_id = str(uuid.uuid4())
#         if session_id not in sessions:
#             sessions[session_id] = {"chat_history": [], "state": {}}

#         user_content = user_message.message
#         sessions[session_id]["chat_history"].append({"sender": "user", "content": user_content})

#         # Create state for workflow
#         state = {
#             "messages": [
#                 HumanMessage(content=msg["content"]) if msg["sender"] == "user"
#                 else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
#                 for msg in sessions[session_id]["chat_history"]
#             ],
#             "chat_history": sessions[session_id]["chat_history"]
#         }

#         # Run workflow
#         workflow_app = create_workflow()
#         response = workflow_app.invoke(state)

#         # Ensure only new bot messages are appended to chat history
#         new_messages = [
#             {"sender": "bot", "content": message.content}
#             for message in response["messages"]
#             if not isinstance(message, HumanMessage)  # Only include non-human messages
#         ]
        
#         sessions[session_id]["chat_history"].extend(new_messages)
#         sessions[session_id]["state"] = state
        
#         last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

#         return JSONResponse(content={"message": last_message["content"]})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import os
# import uuid
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from pydantic_settings import BaseSettings  # Updated import
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, ToolMessage
# from graph import create_workflow

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Interaction Bot v1")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory chat history storage
# chat_history = []

# class UserMessage(BaseModel):
#     message: str

# @app.get("/")
# async def read_root():
#     welcome_message = '''
#     Hi folks! We are Highland Constructors
#     '''
#     return {"message": welcome_message}

# @app.get("/chat-history")
# async def get_chat_history():
#     return chat_history

# @app.post("/send-message")
# async def send_message(user_message: UserMessage):
#     try:
#         session_id = str(uuid.uuid4())  # Generate a new session ID for each request
#         user_content = user_message.message
#         chat_history.append({"session_id": session_id, "sender": "user", "content": user_content})

#         # Create state for workflow
#         state = {
#             "messages": [
#                 HumanMessage(content=msg["content"]) if msg["sender"] == "user"
#                 else ToolMessage(content=msg["content"], tool_call_id=str(uuid.uuid4()))
#                 for msg in chat_history
#             ]
#         }

#         # Run workflow
#         workflow_app = create_workflow()
#         response = workflow_app.invoke(state)

#         # Ensure only new bot messages are appended to chat history
#         new_messages = [
#             {"session_id": session_id, "sender": "bot", "content": message.content}
#             for message in response["messages"]
#             if not isinstance(message, HumanMessage)  # Only include non-human messages
#         ]
        
#         chat_history.extend(new_messages)
        
#         last_message = new_messages[-1] if new_messages else {"content": "No response from bot"}

#         return JSONResponse(content={"message": last_message["content"]})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
