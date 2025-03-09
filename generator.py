import json
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize model and embeddings
model_local = ChatOllama(model='deepseek-r1:latest')
embedding = OllamaEmbeddings(model='nomic-embed-text:latest')

# Function to load PDF documents
def load_docs(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()

# Function to generate interactive conversation between Alex and Mia
def generate_and_save_conversation(folder_path, output_file):
    documents = load_docs(folder_path)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    
    conversation_template = """
   Generate a lively, engaging, and interactive podcast-style conversation between Alex (host) and Mia (guest). The conversation should be dynamic, thought-provoking, and entertaining, covering a mix of fun, insightful, and meaningful topics.
    Guidelines for the Conversation:

        Warm and friendly introduction: Alex welcomes Mia in a natural and inviting way.
        Diverse range of topics: Discuss areas like travel, creativity, personal growth, technology, and thought-provoking ideas.
        Balanced mix of questions: Use “what,” “why,” “how,” “when,” and “which” questions to keep the conversation deep and engaging.
        Spontaneous and interactive: Keep the dialogue fluid, with both Alex and Mia actively responding and exchanging ideas.
        Humor and storytelling: Include natural humor, quick-witted exchanges, and personal experiences to make the conversation immersive.
        Smooth transitions: Ensure a seamless flow between topics rather than abrupt jumps.
        No unnecessary elements: Avoid episode numbers, titles, or narration like (“laughs”)—keep the focus on natural conversation.

    The conversation should be at least 50 lines long, resembling an engaging and authentic podcast discussion.
    {context}
    """
    conversation_prompt = ChatPromptTemplate.from_template(conversation_template)
    
    conversation_chain = (
        {"context": retriever} 
        | conversation_prompt
        | llm
        | StrOutputParser()
    )
    
    conversation = conversation_chain.invoke("Generate an engaging and insightful conversation between Alex and Mia.")
    conversation_lines = conversation.split('\n')
    
    formatted_conversation = {"conversation": []}
    for line in conversation_lines:
        if ':' in line:
            speaker, text = line.split(':', 1)
            formatted_conversation["conversation"].append({"speaker": speaker.strip(), "name": speaker.strip(), "text": text.strip()})
    
    with open(output_file, 'w') as json_file:
        json.dump(formatted_conversation, json_file, indent=4)
    
    print(f"Conversation between Alex and Mia saved to {output_file}")

if __name__ == "__main__":
    folder_path = "Data"
    output_file = "alex_mia_conversation.json"
    generate_and_save_conversation(folder_path, output_file)
