import pinecone
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader

from consts import llm_model_cohere, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")


if __name__ == '__main__':
    #loading the links csv files
    ict = pd.read_csv("Data\ICT_ALL_VIDS - Sheet1.csv")
    youtube_url_list = ict['https://www.youtube.com/watch?v=NeZlyG8FZLQ'].unique()
    youtube_url_list = youtube_url_list.tolist()
    youtube_url_list.extend(["https://www.youtube.com/watch?v=NeZlyG8FZLQ"])
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=128)

    for url in youtube_url_list:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            result = loader.load()
            texts.extend(text_splitter.split_documents(result))
        except Exception as e:
            print(f"Error occurred while transcribing video: {url}. Skipping to the next video.")
            print(f"Error details: {str(e)}")    

    # Create embeddings using cohere so that it can handle multi-lingual languages
    embeddings = CohereEmbeddings(model = llm_model_cohere,cohere_api_key=cohere_api_key)

    #Initialize a pinecone index to store the embeddings
    pinecone.init(
        api_key=pinecone_api_key , # find at app.pinecone.io
        environment=PINECONE_ENVIRONMENT,  # next to api key in console
    )

    #Store the embeddings in a vector database 
    vectordb= Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
    
