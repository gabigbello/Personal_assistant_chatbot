import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
import tiktoken

#load environment variables
load_dotenv()

#load csv document with Q&A
loader = CSVLoader(file_path="electronic_store_faq.csv")
documents = loader.load()

#transform entries from our csv - texts in numbers
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents,embeddings)

#defining function to consult in our dataset based on an entry
def retrieve_info(query):
    similar_response = db.similarity_search(query,k=3)
    return[doc.page_content for doc in similar_response]

llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0125")

template = """
You're a virtual assistant of a electronic store. 
Your function is to respond to e-mails received from potential clients. 
I'll send you some old e-mails sent by our sales team for you to use as a model.

Follow all these rules below:
1/ Always greet customers politely.
2/ Maintain a professional tone.
3/ Use respectful and courteous language.
4/ Avoid technical jargon; use simple and clear language.
5/ If unsure about an answer, direct customers to contact a human representative.
6/ Ensure customer data is handled in compliance with privacy laws and company policies.
7/ Avoid long delays in providing information.
8/ Aim to resolve customer issues efficiently.
9/ Offer practical solutions and alternatives when possible.
10/ Offer additional help if it seems the customer may need it (e.g., related product suggestions, help with the checkout process).

Here is a message received from a new client.
{message}

Write the best answer that I could send to this potential client:
"""

prompt = PromptTemplate(
    input_variables=["message"],
    template = template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generaste_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message)
    return response

def main():
    st.set_page_config(
        page_title="Virtual assistant - Electronic Store X20",page_icon=":owl:")
    st.header("Virtual chat")
    message=st.text_area("Your question")

    if message:
        st.write("Generating an answer based on our best practices...")

        result = generaste_response(message)

        st.info(result)

if __name__ == '__main__':
    main()


generaste_response("What are the brands that I can find in your store?")
