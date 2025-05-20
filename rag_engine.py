import os
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.document_loader import load_documents, process_documents

class FitnessRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.7
        )
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the conversation chain
        self.chain = self._create_conversation_chain()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store with fitness knowledge"""
        # Check if vector store already exists
        if os.path.exists("vectorstore"):
            return Chroma(
                persist_directory="vectorstore",
                embedding_function=self.embeddings
            )
        
        # Create new vector store
        documents = self._load_fitness_knowledge()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="vectorstore"
        )
        return vector_store
    
    def _create_conversation_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational chain with custom prompts"""
        prompt_template = """You are an expert fitness coach with deep knowledge of exercise science, nutrition, and wellness. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always maintain a supportive and encouraging tone, and prioritize safety in your advice.
        
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
    
    def _load_fitness_knowledge(self) -> List[Dict]:
        """Load fitness knowledge from various sources"""
        # Load documents from the fitness knowledge directory
        documents = load_documents("data/fitness_knowledge")
        # Process the documents
        processed_documents = process_documents(documents)
        return processed_documents
    
    def get_response(self, query: str) -> str:
        """Get a response from the RAG system"""
        try:
            response = self.chain.invoke({"question": query})
            return response["answer"]
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question." 