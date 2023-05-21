import gradio as gr
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from consts import llm_model_cohere, llm_model_openai, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")





embeddings = CohereEmbeddings(model = llm_model_cohere,cohere_api_key=cohere_api_key)

db = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
memory = ConversationBufferMemory(memory_key ='chat_history', 
                                  return_messages= False,)

prompt_template = """
You have been a profitable trader for a very long time. Whenever someone asks a question, please answer them with a clear plan 
of how to use what they have asked to be profitable as a trader. Your answer should be long, detailed , specific , accurate and smart.
Aso you should be very encouraging and motivating in your responses. If possible provide your sources to the answers you provide. If the text doesn't contain the answer, reply that the answer is not available.


Text: {context}
Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
# chain = load_qa_chain(Cohere(model="command-xlarge-nightly", temperature=0), chain_type="stuff", prompt=PROMPT)
qa = ConversationalRetrievalChain.from_llm(
    llm = OpenAI(temperature=0, max_tokens=-1,openai_api_key=openai_api_key),
    chain_type = 'stuff',
    retriever = db.as_retriever(),
    memory=memory,
    get_chat_history= lambda h: h,
    verbose = True,
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id='chatbot').style(height=500)
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        print(history)
        bot_message = qa.run({'question': history[-1][0], 'chat_history': history[:-1]})
        history[-1][1]= bot_message
        return history
    msg.submit(user, [msg, chatbot],[msg, chatbot], queue=False).then(bot,chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()


