import os
import logging
from typing import List
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: SecretStr
    OPENAI_API_KEY: SecretStr
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    LANGCHAIN_TRACING_V2: bool = False  # Assuming it is a boolean
    LANGCHAIN_API_KEY: SecretStr  # Adding this as it seems to be a secret key
    LANGCHAIN_PROJECT: str  # Adding this to the model

    class Config:
        env_file = ".env"

# Initialize settings
try:
    settings = Settings()
except ValidationError as e:
    logger.error("Configuration error: %s", e)
    raise

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    temperature=0.0,
    model_name='gpt-4o',
    openai_api_key=settings.OPENAI_API_KEY.get_secret_value()
)

def init_neo4j_graph():
    try:
        graph = Neo4jGraph(url=settings.NEO4J_URI, username=settings.NEO4J_USERNAME, password=settings.NEO4J_PASSWORD.get_secret_value())
        if graph:
            logger.info("Graph connected successfully")
        else:
            logger.error("Graph connection failed")
        return graph
    except Exception as e:
        logger.exception("Failed to connect to Neo4j Graph: %s", str(e))
        raise

graph = init_neo4j_graph()

def init_vector_index(graph):
    embeddings = OpenAIEmbeddings(
    model=settings.OPENAI_EMBEDDING_MODEL,
    openai_api_key=settings.OPENAI_API_KEY.get_secret_value()
)
    try:
        vector_index = Neo4jVector.from_existing_graph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD.get_secret_value(),
            embedding=embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        logger.info("Vector index created successfully")
        return vector_index
    except Exception as e:
        logger.exception("Failed to create Neo4j vector index: %s", str(e))
        raise

vector_index = init_vector_index(graph)

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

def generate_full_text_query(input: str) -> str:
    """
    Generates a full text query by removing Lucene characters from the input string and 
    splitting it into words. Each word is then appended to the query string with a 
    fuzzy match operator (~2) and separated by the AND operator. 

    Args:
        input (str): The input string to generate the full text query from.

    Returns:
        str: The generated full text query.
    """
    words = [el for el in remove_lucene_chars(input).split() if el]
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    return full_text_query

def structured_retriever(question: str) -> str:
    """
    Retrieves structured data based on the provided question and returns the formatted output.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ])

    entity_chain = prompt | llm.with_structured_output(Entities)

    try:
        entities = entity_chain.invoke({"question": question})
        result = ""
        for entity in entities:
            # response = graph.query(
            #     """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
            #        YIELD node,score
            #        CALL {
            #          WITH node
            #          MATCH (node)-[r:!MENTIONS]->(neighbor)
            #          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            #          UNION ALL
            #          WITH node
            #          MATCH (node)<-[r:!MENTIONS]-(neighbor)
            #          RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            #        }
            #        RETURN output LIMIT 50
            #     """,
            #     {"query": generate_full_text_query(entity)},
            # )
            # result += "\n".join([el['output'] for el in response])

            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit: 5})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r:MENTIONS]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r:MENTIONS]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])


        return result
    except Exception as e:
        logger.exception("Error during structured retrieval: %s", str(e))
        return "An error occurred during structured retrieval."

def retriever(question: str) -> str:
    """
    Retrieves structured and unstructured data based on a given question.

    Args:
        question (str): The question to retrieve data for.

    Returns:
        str: The retrieved structured and unstructured data in the following format:
             "Structured data:\n{structured_data}\nUnstructured data:\n{unstructured_data}"
             If an error occurs during data retrieval, returns "An error occurred during data retrieval."

    Raises:
        Exception: If an error occurs during data retrieval.
    """
    try:
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:\n{structured_data}\nUnstructured data:\n{"#Document ".join(unstructured_data)}"""
        return final_data
    except Exception as e:
        logger.exception("Error during data retrieval: %s", str(e))
        return "An error occurred during data retrieval."
