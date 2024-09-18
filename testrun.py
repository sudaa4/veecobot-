
import os
import annoy 
from flask import Flask, request, render_template, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Annoy
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Initialize Flask app
app = Flask(__name__)

# Ensure required packages are installed
try:
    import langchain
except ImportError:
    os.system('pip install langchain openai tiktoken faiss-cpu')

# Set the OpenAI API key
api_key = input("Please enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key

# Check that the environment variable was set correctly
print("OPENAI_API_KEY has been set!")

# path to the filepython -m pip show annoy
file_path = r"C:\Users\USER\Downloads\introduction.txt"

# Load the file using TextLoader
loader = TextLoader(file_path, encoding="utf-8")
data = loader.load()

# Split the text into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create embeddings and a vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Annoy.from_documents(data, embedding=embeddings, n_trees=10)

# Initialize the language model and memory
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create a conversational retrieval chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Now the setup is complete, and you can interact with the conversation chain
print("Hi I am Veecobot. How can I help you ? ")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    # Get the response from the conversation chain
    response = conversation_chain({"question": query})
    
    # Print the response
    print(f"Bot: {response['answer']}") 



    @app.route('/')
 def index():
    return render_template('index.html')  # Serve the chatbot interface

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question')
    if user_input:
        # Get the response from the chatbot
        response = conversation_chain({"question": user_input})
        return jsonify({"answer": response['answer']})
    return jsonify({"answer": "Please ask a question."})

if __name__ == '__main__':
    app.run(debug=True)