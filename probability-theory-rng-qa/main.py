import os
import telebot
from dotenv import load_dotenv
import ai
import sympy

load_dotenv()

bot_token = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(bot_token)

qas = {}

def render_latex(filename, expression):
  preamble = '\\documentclass[varwidth]{standalone}' \
    '\\usepackage[english, russian]{babel}' \
    '\\usepackage{mathtext}' \
    '\\usepackage[utf8]{inputenc}' \
    '\\usepackage{amsmath,amsfonts,amssymb,amsthm,mathtools}' \
    '\\begin{document}'

  sympy.preview(
    expression,
    viewer='file',
    filename=filename,
    euler=False,
    dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"],
    preamble=preamble,
  )

@bot.message_handler(commands=['start'])
def start(message):
  bot.send_chat_action(chat_id=message.chat.id, action='typing')

  qa = ai.get_qa()

  bot.send_message(message.chat.id, qa['query'])

  qas[message.chat.id] = qa

@bot.message_handler()
def root(message):
  qa = qas[message.chat.id]
  is_correct_answer = ai.check_answer(qa, message.text)
  model_answer = qa['answer']

  if is_correct_answer:
    bot.send_message(message.chat.id, 'Правильно! ❤️ \nМой ответ был:')
  else:
    bot.send_message(message.chat.id, 'Неправильно 😔 Правильный ответ:')

  try:
    filename = f'./images/{message.chat.id}.png'
    render_latex(filename, model_answer.replace('\\\\', '\\'))
    f = open(filename, 'rb')
    bot.send_photo(message.chat.id, f)
  except:
    bot.send_message(message.chat.id, model_answer)

  qa = ai.get_qa()

  bot.send_message(message.chat.id, qa['query'])

  qas[message.chat.id] = qa

bot.infinity_polling()
