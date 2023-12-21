import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data():    
    tokenizer = AutoTokenizer.from_pretrained("SheilaCXY/DialoGPT-RickBot")
    model = AutoModelForCausalLM.from_pretrained("SheilaCXY/DialoGPT-RickBot")
    return tokenizer, model

tokenizer, model = load_data()




def chat(message, history):
  history = history or []
  new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
  
  if len(history) > 0 and len(history) < 2:
    for i in range(0,len(history)):
      encoded_message = tokenizer.encode(history[i][0] + tokenizer.eos_token, return_tensors='pt')
      encoded_response = tokenizer.encode(history[i][1] + tokenizer.eos_token, return_tensors='pt')
      if i == 0:
        chat_history_ids = encoded_message
        chat_history_ids = torch.cat([chat_history_ids,encoded_response], dim=-1)
      else:
        chat_history_ids = torch.cat([chat_history_ids,encoded_message], dim=-1)
        chat_history_ids = torch.cat([chat_history_ids,encoded_response], dim=-1)
      
      bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

  elif len(history) >= 2:
      for i in range(len(history)-1, len(history)):
        encoded_message = tokenizer.encode(history[i][0] + tokenizer.eos_token, return_tensors='pt')
        encoded_response = tokenizer.encode(history[i][1] + tokenizer.eos_token, return_tensors='pt')
        if i == (len(history)-1):
          chat_history_ids = encoded_message
          chat_history_ids = torch.cat([chat_history_ids,encoded_response], dim=-1)
        else:
          chat_history_ids = torch.cat([chat_history_ids,encoded_message], dim=-1)
          chat_history_ids = torch.cat([chat_history_ids,encoded_response], dim=-1)
      
      bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
  
  elif len(history) == 0:
    bot_input_ids =  new_user_input_ids

  chat_history_ids = model.generate(bot_input_ids, max_length=1000, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
  response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
  history.append((message, response))

  return history, history

title = "DialoGPT fine-tuned on DailyDialog"
description = "Gradio demo for open-domain dialog using DialoGPT. Model was fine-tuned on the DailyDialog multi-turn dataset"
iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
    title=title,
    description=description
)
iface.launch(debug=True)
