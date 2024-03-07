import os
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

auth_data = os.getenv('GIGACHAT_AUTH_DATA')
llm = GigaChat(credentials=auth_data, verify_ssl_certs=False)

retriever = WikipediaRetriever()

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

chat_history = []

while True:
  question = input('You: ')

  if not question:
    break

  result = qa({'question': question, 'chat_history': chat_history})
  answer = result['answer']

  chat_history.append((question, answer))
  print(f'AI: {answer}\n')
