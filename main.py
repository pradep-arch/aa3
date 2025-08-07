# pip install llama-index llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq pandas matplotlib seaborn

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


MODEL_NAME = "llama-3.1-70b-versatile"  # Updated to a supported Groq model
API_KEY = "YOUR_API_KEY"  # Replace with your Groq API key
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DATA_PATH = "./data/"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 512

def get_llm(model_name, api_key):
    return Groq(model=model_name, api_key=api_key, temperature=0.5)

def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def analyze_and_respond(review: str) -> str:
    """Analyzes sentiment of a review and generates a polite response."""
    
    prompt = f"Analyze the sentiment of this review (positive, negative, or neutral) and generate a short, polite response:\nReview: {review}"
    response = Settings.llm.complete(prompt).text
    
   
    sentiment = "positive" if "positive" in response.lower() else "negative" if "negative" in response.lower() else "neutral"
    reply = response.split("Response:")[-1].strip() if "Response:" in response else "Thank you for your feedback!"
    return f"Sentiment: {sentiment}\nResponse: {reply}"


def plot_sentiment_trend(start_date: str, end_date: str) -> str:
    """Generates a bar plot of sentiment trends for a given date range."""
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, "restaurant_reviews.csv"))
        df['date'] = pd.to_datetime(df['date'])
        
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
       
        mask = (df['date'] >= start) & (df['date'] <= end)
        filtered_df = df[mask]
        

        sentiment_counts = filtered_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336', '#FFC107'])
        plt.title(f"Sentiment Trends from {start_date} to {end_date}")
        plt.xlabel("Date")
        plt.ylabel("Number of Reviews")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(DATA_PATH, f"sentiment_trend_{start_date}_to_{end_date}.png")
        plt.savefig(output_path)
        plt.close()
        
        return f"Plot saved at: {output_path}"
    except Exception as e:
        return f"Error generating plot: {str(e)}"


response_tool = FunctionTool.from_defaults(fn=analyze_and_respond)
visualization_tool = FunctionTool.from_defaults(fn=plot_sentiment_trend)


initialize_settings()
agent = ReActAgent.from_tools(
    tools=[response_tool, visualization_tool],
    llm=Settings.llm,
    verbose=True
)


if __name__ == "__main__":
    # Test Feedback Response Agent
    print("Testing Feedback Response Agent")
    response = agent.chat("The food was amazing and the service was quick!")
    print(response)
    
    # Test Sentiment Visualization Agent
    print("\n Testing Sentiment Visualization Agent")
    response = agent.chat("Plot sentiment trends from 2025-08-01 to 2025-08-04")
    print(response)