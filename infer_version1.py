# 此版本未实现badcase和commends功能
from transformers import MarianMTModel, MarianTokenizer
import gradio as gr
import os
import json
import time
import requests

# Load config file
config_file = 'config.json'
with open(config_file, 'r') as f:
    config_data = json.load(f)

# Helper functions
def get_lang_pairs(config_data):
    source_languages = list(config_data.keys())
    lang_pairs = {src: list(targets.keys()) for src, targets in config_data.items()}
    return source_languages, lang_pairs

def get_max_version_folder(langpair_folder):
    if not os.path.exists(langpair_folder) or not os.path.isdir(langpair_folder):
        return None
    file_list = []
    try:
        for folder in os.listdir(langpair_folder):
            if folder.startswith("checkpoint-"):
                try:
                    version = int(folder.split("-")[1])
                    file_list.append((version, folder))
                except (IndexError, ValueError):
                    continue
    except Exception as e:
        print(f"Error accessing langpair_folder: {e}")
        return None
    if len(file_list) == 0:
        return None
    else:
        max_version_folder = max(file_list)[1]
        return os.path.join(langpair_folder, max_version_folder)

def check_model_availability(src_lang, tgt_lang):
    if src_lang in config_data and tgt_lang in config_data[src_lang]:
        return config_data[src_lang][tgt_lang], True
    return None, False

def delayed_language_check(src_lang, tgt_lang):
    time.sleep(0.5)
    model_path, exists = check_model_availability(src_lang, tgt_lang)
    if exists:
        return "success", model_path
    else:
        return "failure", None

def load_model_and_tokenizer(model_path):
    model_path = get_max_version_folder(model_path)
    if model_path is None or not os.path.isdir(model_path):
        return None, None, "failure"
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    return model, tokenizer, "success"

def perform_translation(text, model, tokenizer):
    if model is None or tokenizer is None:
        return "请先选择语言对并加载模型！"
    if not text:
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def update_target_dropdown(src_lang): 
    if isinstance(src_lang, list):
        src_lang = src_lang[0]
    if src_lang in lang_pairs:
        return gr.update(choices=lang_pairs[src_lang], value=None)
    return gr.update(choices=[], value=None)

def check_and_load_model(src_lang, tgt_lang):
    if src_lang and tgt_lang:
        status, model_path = delayed_language_check(src_lang, tgt_lang)
        if model_path:
            model, tokenizer, load_status = load_model_and_tokenizer(model_path)
            input_box_state = gr.update(
                interactive=True if load_status == "success" else False,
                placeholder="请输入要翻译的文本..." if load_status == "success" else "请先选择源语言和目标语言。"
            )
            return model, tokenizer, input_box_state
        else:
            return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。")
    return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。")

def perform_google_translation(text, src_lang, tgt_lang):
    if not text:
        return ""
    try:
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            "q": text,
            "source": src_lang,
            "target": tgt_lang,
            "key": ""  # Replace with your Google API key
        }
        response = requests.get(url, params=params)
        response_data = response.json()
        if "data" in response_data and "translations" in response_data["data"]:
            return response_data["data"]["translations"][0]["translatedText"]
        else:
            return "Google API 错误，请检查请求。"
    except Exception as e:
        return f"Google API 调用失败: {e}"


source_languages, lang_pairs = get_lang_pairs(config_data)


with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>机器翻译界面</h1>")
    
    with gr.Row():
        with gr.Column():
            src_lang_dropdown = gr.Dropdown(choices=source_languages, label="源语言", interactive=True)
            input_text = gr.Textbox(label="输入文本", lines=5, placeholder="请先选择源语言和目标语言。", interactive=False)
        
        with gr.Column():
            tgt_lang_dropdown = gr.Dropdown(choices=[], label="目标语言", interactive=True)
            output_text = gr.Textbox(label="翻译结果", lines=5, interactive=False)
        
    with gr.Row():
        google_translate_button = gr.Button("使用Google翻译", elem_id="google_translate_button")
    
    model_var = gr.State()
    tokenizer_var = gr.State()

    src_lang_dropdown.change(fn=update_target_dropdown, inputs=src_lang_dropdown, outputs=tgt_lang_dropdown)

    tgt_lang_dropdown.change(
        fn=check_and_load_model,
        inputs=[src_lang_dropdown, tgt_lang_dropdown],
        outputs=[model_var, tokenizer_var, input_text]
    )

    input_text.change(fn=perform_translation, inputs=[input_text, model_var, tokenizer_var], outputs=output_text)

    google_translate_button.click(fn=perform_google_translation, inputs=[input_text, src_lang_dropdown, tgt_lang_dropdown], outputs=output_text)

# Add custom CSS for styling
demo.css = """
#google_translate_button {
    width: 300px;
    margin: 20px auto;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
    color: blue;
    background-color: white;
    border: 2px solid blue;
    border-radius: 8px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

#google_translate_button:hover {
    background-color: blue;
    color: blue;
}

#google_translate_button:active {
    background-color: darkblue;
    color: white;
}
"""
demo.launch(server_name="0.0.0.0",server_port=7999)
