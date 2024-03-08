import os
import random
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

load_dotenv()

loader = WikipediaLoader(query='star wars', load_max_docs=20)
docs = loader.load()

auth_data = os.getenv('GIGACHAT_AUTH_DATA')
llm = GigaChat(credentials=auth_data, verify_ssl_certs=False, temperature=1)
gen_chain = QAGenerateChain.from_llm(llm)

eval_chain = QAEvalChain.from_llm(llm)

while True:
  doc = random.choice(docs)

  qa = gen_chain(doc)['qa_pairs']
  question = qa['query']
  correct_answer = qa['answer']

  print(f'Question: {question}')

  user_answer = input('Answer: ')

  graded_output = eval_chain.evaluate([qa], [{'result': user_answer}])[0]['results']

  if graded_output == 'CORRECT':
    print('You\'re right!')
  else:
    print(f'No, {correct_answer}')
