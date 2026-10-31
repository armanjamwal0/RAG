from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


loader = DirectoryLoader(
    path='data/pdf',
    glob='**/*.pdf',
    loader_cls=PyMuPDFLoader,
    use_multithreading=True,
    show_progress=True
)

doc = loader.load()

## convert into chunks 
chunk = RecursiveCharacterTextSplitter( ## sementic chunking
    chunk_size=800,
    chunk_overlap=100
)

chunks = chunk.split_documents(doc) 

## Embeddings Model 
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",encode_kwargs={"normalize_embeddings": True},model_kwargs={'truncate_dim':768})

## store embeddings in vector db
vector_store = Chroma(
    collection_name='rag_data',
    embedding_function=embeddings,
    persist_directory='chroma_db'
)

vector_store.add_documents(chunks)
print(f'Stored embeddings in vector db and Total chunks : {len(chunks)}')
print('DONE...........................................................')