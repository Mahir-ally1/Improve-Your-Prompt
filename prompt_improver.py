from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import tiktoken
import os
import llm_secrets



# Retrieval

# Load Markdown text data
def read_markdown(markdown_path):
    loader = UnstructuredMarkdownLoader(markdown_path, mode='elements')  # Every line is considered as a single document
    data = loader.load()
    return data

# Load all text (.txt) files
def read_docs(docs_folder):
    text_files_path = []
    documents = []
    for file in os.listdir(docs_folder):
        if file.endswith(".txt"):
            text_files_path.append(os.path.join(docs_folder, file))
            loader = TextLoader(os.path.join(docs_folder, file))
            doc = loader.load()
            documents.append(doc[0].page_content)  # Each file is considered as a single document
    return documents

# Split documents semantically
def semantic_split(llm_api_key, docs):
    embeddings = OpenAIEmbeddings(api_key=llm_api_key, model='text-embedding-3-small')  # Keeping the original model
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="gradient",
        # breakpoint_threshold_type="fixed",
        breakpoint_threshold_amount=0.8
    )
    
    all_chunks = []
    chunks = semantic_splitter.split_text(doc)
    all_chunks.extend(chunks)
    
    return all_chunks

# Recursive character text split
def text_split(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "], 
        chunk_size=500,
        chunk_overlap=100
    )

    all_chunks = []
    chunks = text_splitter.split_text(doc)
    all_chunks.extend(chunks)
    
    return all_chunks

def token_split(doc):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    token_splitter = TokenTextSplitter(encoding_name=encoding.name,
        chunk_size=10,
        chunk_overlap=2)
    all_chunks = []
    chunks = token_splitter.split_text(doc)
    all_chunks.extend(chunks)
    
    return all_chunks       

# Read Text File
docs_folder = 'context_guidelines'
docs = read_docs(docs_folder)

# Semantically splitting text documents into chunks
llm_api_key = llm_secrets.llm_api_key


# recursive_doc_chunks = []
# semantic_doc_chunks = []
# token_doc_chunks = []
# for doc in docs:
#     # recursive_chunks = text_split(doc)
#     # recursive_doc_chunks.extend(recursive_chunks)
#     # semantic_chunks = semantic_split(llm_api_key, doc)
#     # semantic_doc_chunks.extend(semantic_chunks)
#     token_chunks = token_split(doc)
#     token_doc_chunks.extend(token_chunks)


# # for i, chunk in enumerate(recursive_doc_chunks):
# #   print(f"Chunk {i+1}: {chunk} (Length: {len(chunk)})")


# # for i, chunk in enumerate(semantic_doc_chunks):
# #   print(f"Chunk {i+1}: {chunk} (Length: {len(chunk)})")



# # Print the token chunks
# for i, chunk in enumerate(token_doc_chunks):
#     print(f"Chunk {i+1}: {chunk} (Length: {len(chunk)})")



# Read Markdown file
markdown_doc_chunks = []
markdown_path = "context_guidelines/README.md"
markdown_data = read_markdown(markdown_path)
print(f"Number of documents: {len(markdown_data)}\n")


for i in range(len(markdown_data)):
    markdown_doc_chunks.extend(markdown_data[i].page_content)


# Embedding

embedding_model = OpenAIEmbeddings(
    api_key=llm_api_key,
    model="text-embedding-3-small"
)

vector_store = Chroma.from_texts(
    texts = markdown_doc_chunks, # only using markdown chunks for now
    embedding = embedding_model

)

retriever = vector_store.as_retriever (
    search_type = "similarity",
    search_kwargs = {"k": 10}

)

file_path = 'prompts/instruction_prompt.txt'

# Open the file and read its contents into a string
with open(file_path, 'r', encoding='utf-8') as file:
    instruction = file.read()

prompt = ChatPromptTemplate.from_template(instruction)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=llm_api_key, temperature=0)

chain = (
    {"context":retriever, "user_prompt": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("Write me an essay for my history class?")
print(result)