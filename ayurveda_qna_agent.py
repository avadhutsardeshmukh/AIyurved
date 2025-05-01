#from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
#os.environ['USER_AGENT'] = 'myagent'
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_unstructured import UnstructuredLoader 

'''
#Loads a single pdf doc, splits it into chunks of desired size and returns the chunks
@param filepath: path to the directory containing all the pdfs
@return : List of chunks 
'''
def doc_loader_splitter(filepath, chunksize):
    loader = PyPDFLoader(
            file_path = filepath,
            #strategy="fast"
            extract_images=False
            )
    docs = []
    docs_lazy = loader.lazy_load()
    # async variant:
    # docs_lazy = await loader.alazy_load()
    for doc in docs_lazy:
        docs.append(doc)
    print(len(docs))
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunksize, chunk_overlap=0
            )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits
'''
#Loads all the documents in the directory, splits them and returns the chunks
@param filepath: path to the directory containing all the pdfs
@return : Flattened list of chunks 
'''
def multi_doc_loader_splitter(filepath, chunksize):
    all_docs = []
    if (not os.path.isdir(filepath)):
        print('Provided path is not a directory. \
        If you are looking to load a single pdf, call doc_loader_splitter() instead')
        exit()
    for ff in os.listdir(filepath):
        if(os.path.splitext(ff)[1] == '.pdf'):
            all_docs = all_docs + doc_loader_splitter(os.path.join(filepath, ff), chunksize)
    return all_docs

'''
#Creates a vector store and stores the input docs into the store, using an embedding method.
@param docs : the list of chunks created from all the documents
@param embedding : the embedding method to be used to create the vector db
'''
def create_vector_store(chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    vector_store = SKLearnVectorStore.from_documents(documents=chunks, embedding=hf)
    #query="What are some herbs mentioned in charaka samhita that have antimicrobial properties against rhinitis"
    #docs = vector_store.similarity_search(query)
    return vector_store.as_retriever(k=5)

# Define the prompt template for the LLM
def create_chain(ll_model):
    prompt = PromptTemplate(
    template="""You are an expert in Ayurveda, with knowledge of texts 
    such as Bhavprakash Nighantu and Sharangdhar Samhita.
    Please help me with some information regarding various herbal plants using the following documents as background knowledge.
    You might as well use your earlier training to infer answers not directly present in the provided documents.
    To the extent possible, provide concise answers.
    Question: {question}
    Documents: {context}
    Answer:
    """,
    input_variables=["question", "context"],
    )
    llm = ChatOllama(
        model=ll_model,
        temperature=0.1
    )
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"context": doc_texts,"question": question})
        return answer

if __name__ == '__main__':
    #print("Hello")
    import sys
    filepath = sys.argv[1]
    print(filepath)
    split_docs = multi_doc_loader_splitter(filepath, 1000)
    print(len(split_docs))
    
    retriever = create_vector_store(split_docs)
    rag_chain = create_chain('llama3.2')
    print('Created RAG chain')

    # Initialize the RAG application
    rag_application = RAGApplication(retriever, rag_chain)
    # Example usage
    question = "Which plants are possess antimicrobial properties against rhinitis, cold or cough?"
    answer = rag_application.run(question)
    print("Question:", question)
    print("Answer:", answer)
