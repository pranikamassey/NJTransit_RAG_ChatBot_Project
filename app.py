# app.py
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/query"

def ask(question):
    payload = {"question": question}
    r = requests.post(API_URL, json=payload)
    if r.status_code == 200:
        return r.json()["answer"]
    return f"Error {r.status_code}: {r.text}"

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Type your question…"),
    outputs="text",
    title="NJ Transit AccessLink Bot",
    description="Ask anything about the Access Link guidelines."
)

if __name__ == "__main__":
    demo.launch()
