from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
load_dotenv()

# pdf loading
loader = PyPDFDirectoryLoader("pdfs")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 300)
final_documents = text_splitter.split_documents(documents)



# Embeddings using Hugging Face
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)


# vectore store
vectorstore = FAISS.from_documents(final_documents,huggingface_embeddings)
vectorstore.save_local("faiss_index")
