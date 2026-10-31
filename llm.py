from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"truncate_dim": 768}
)

# Connect to EXISTING ChromaDB
vector_store = Chroma(
    collection_name="rag_data",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# Retrieve documents
query = "What is Private Filtering ?"

results = vector_store.similarity_search(
    query,
    k=3
)

doc = '\n\n'.join([doc.page_content for doc in results])

# print(doc)
# for doc in results:
#     print(doc.page_content)
#     print(doc.metadata)

# Answer the question using ONLY the context below.
# If the answer is not in the context, say "I don't know."

# Context:
# <chunk 1>
# <chunk 2>
# ... (5 chunks joined together)

# Question: <user's question>

parser = StrOutputParser()

prompt = PromptTemplate(
    template= 
    """
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context: {context}

Question: {question}

""",
input_variables=['context','question']
)
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

reslut_chain = prompt | model | parser

result = reslut_chain.invoke({'context':doc,'question':query})
print(result)

