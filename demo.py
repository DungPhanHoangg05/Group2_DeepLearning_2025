import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# Load model 
model_name = "gpt2_sft" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

# Hàm sinh phản hồi
def chat(prompt, history):
    full_prompt = ""
    for user_msg, bot_msg in history:
        full_prompt += f"Prompt: {user_msg}\nResponse: {bot_msg}\n"
    full_prompt += f"Prompt: {prompt}\nResponse:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_length=512,
          num_beams=5,
          early_stopping=True,
          no_repeat_ngram_size=3,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id
      )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("Response:")[-1].strip()
    return response

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🧑‍⚖️ Legal Chatbot Demo (GPT-2)")
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(label="Your Prompt", placeholder="Ask a legal question...")
        send_btn = gr.Button("Send")
    state = gr.State([])

    def respond(user_input, history):
        response = chat(user_input, history)
        history.append((user_input, response))
        return history, history

    send_btn.click(respond, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(respond, inputs=[msg, state], outputs=[chatbot, state])

demo.launch(share=True)