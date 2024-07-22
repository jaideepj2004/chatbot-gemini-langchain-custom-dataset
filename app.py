from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

app = Flask(__name__)
load_dotenv()

# Set up Google Generative AI
GOOGLE_API_KEY = os.getenv('AIzaSyCwmGOl-1oTUJgQc5dgShxJItP1bf8q_7w')
genai.configure(api_key=GOOGLE_API_KEY)

# Load and process documents
loader = PyPDFDirectoryLoader("pdfs", load_hidden=True)
data = loader.load_and_split()
context = "\n\n".join(str(p.page_content) for p in data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
texts = text_splitter.split_text(context)

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

# Create prompt template
prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    docs = vector_index.get_relevant_documents(question)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return jsonify({'response': response['output_text']})

if __name__ == '__main__':
    app.run(debug=True)
