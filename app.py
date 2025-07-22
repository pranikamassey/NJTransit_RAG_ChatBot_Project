# app.py
import os
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/query"

def ask(history, user_input):
    """Appends dicts to the message history instead of tuples."""
    if not user_input:
        return history, ""  # nothing to do
    
    # 1) Add the user message
    history = history + [{"role": "user", "content": user_input}]
    
    # 2) Call your backend
    r = requests.post(API_URL, json={"question": user_input})
    if r.status_code == 200:
        bot_content = r.json().get("answer", "")
    else:
        bot_content = f"Error {r.status_code}: {r.text}"
    
    # 3) Add the assistant message
    history = history + [{"role": "assistant", "content": bot_content}]
    
    # 4) Clear input box
    return history, ""

# Custom CSS
custom_css = """
#chatbot .message.user { 
    background-color: #0052A5 !important; 
    color: #fff !important; 
}
#chatbot .message.assistant {
    background-color: #E5E5E5 !important;
    color: #333 !important;
}
.gradio-container {
    background-color: #F8F9FA;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("## 🚌 NJ Transit AccessLink Chatbot\nAsk anything about the Access Link paratransit program!")
    
    # Note type="messages" now expects dicts with role/content
    chatbot = gr.Chatbot(elem_id="chatbot", label="AccessLink Bot", height=500, type="messages")
    
    with gr.Row():
        txt = gr.Textbox(placeholder="Type your question and press Enter", show_label=False, container=False)
    
    # wire up Enter key
    txt.submit(ask, [chatbot, txt], [chatbot, txt])
    
    # clear button resets history to empty list of dicts
    clear = gr.Button("Clear Chat")
    clear.click(lambda: ([], ""), [], [chatbot, txt])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7861))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
