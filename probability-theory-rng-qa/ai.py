import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.output_parsers.regex import RegexParser
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import random
from dotenv import load_dotenv

load_dotenv()

dataset_base_path = './dataset'
paths = [os.path.join(dataset_base_path, filename) for filename in os.listdir(dataset_base_path)]

loader = DirectoryLoader('./dataset/', '*.tex')
docs = loader.load()

index = VectorstoreIndexCreator(
  vectorstore_cls=DocArrayInMemorySearch,
).from_loaders([loader])

def get_llm(temperature=0.7):
  return ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)

def get_gen_chain():
  llm = get_llm()
  template = """Ты - учитель, которому предстоит составить вопросы для контрольной работы.
  Предоставив следующий документ, составь вопрос и ответ на основе этого документа.

  Пример формата:
  <Begin Document>
  ...
  <End Document>
  ВОПРОС: здесь вопрос
  ОТВЕТ: здесь ответ

  Эти вопросы должны быть подробными и основываться непосредственно на информации, содержащейся в документе. Начинай!

  <Begin Document>
  {doc}
  <End Document>"""
  prompt = PromptTemplate(input_variables=['doc'], template=template)
  output_parser = RegexParser(
    regex='ВОПРОС: (.*?)\n+ОТВЕТ: ((.|\n)*)',
    output_keys=['query', 'answer'],
  )

  gen_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=output_parser,
  )
  gen_chain.output_key = 'qa_pairs'

  return gen_chain

def get_eval_chain():
  llm = get_llm(temperature=0)
  eval_chain = QAEvalChain.from_llm(llm)
  return eval_chain

gen_chain = get_gen_chain()
eval_chain = get_eval_chain()

def get_qa():
  doc = random.choice(docs)

  while True:
    try:
      qas = gen_chain.apply_and_parse([{'doc': doc}])
      return qas[0]['qa_pairs']
    except Exception as e:
      print(e)
      pass

def check_answer(qa, user_answer):
  while True:
    try:
      graded_outputs = eval_chain.evaluate([qa], [{'result': user_answer}])
      return graded_outputs[0]['results'] == 'CORRECT'
    except Exception as e:
      print(e)
      pass

