import os
from langchain.chat_models.gigachat import GigaChat
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

auth_data = os.getenv('GIGACHAT_AUTH_DATA')
llm = GigaChat(credentials=auth_data, verify_ssl_certs=False)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=300)

conversation = ConversationChain(llm=llm, memory=memory)

while True:
  user_input = input('Human: ')
  response = conversation.predict(input=user_input)

  print(f'AI: {response}')
