# RAG Fitness Coach Chatbot

A conversational AI fitness coach that uses Retrieval-Augmented Generation (RAG) to provide personalized fitness advice, workout plans, and nutrition guidance.

## Features

- Personalized workout recommendations
- Nutrition advice
- Form guidance for exercises
- Progress tracking suggestions
- Custom knowledge base of fitness information

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `rag_engine.py`: RAG implementation and vector store management
- `data/`: Knowledge base documents and fitness resources
- `utils/`: Helper functions and utilities
