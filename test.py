from transformers import MarianMTModel, MarianTokenizer
import gradio as gr
import os
import json
import time
import requests
import openai
from loguru import logger
from datetime import datetime
# 加载 config.json 和 mapping.jsonl
config_file = 'config.json'
mapping_file = 'mapping.jsonl'

with open(config_file, 'r') as f:
    config_data = json.load(f)

# 加载语言映射
def load_language_mapping(mapping_file):
    language_mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            mapping = json.loads(line.strip())
            language_mapping.update(mapping)
    # 创建反向映射：中文 -> 英文
    reverse_mapping = {v: k for k, v in language_mapping.items()}
    return language_mapping, reverse_mapping

language_mapping, reverse_language_mapping = load_language_mapping(mapping_file)

# 获取语言对，界面显示中文，内部仍使用英文缩写
def get_lang_pairs_with_mapping(config_data, language_mapping):
    source_languages = list(config_data.keys())
    lang_pairs = {src: list(targets.keys()) for src, targets in config_data.items()}

    # 显示为中文
    source_languages_display = [language_mapping.get(src, src) for src in source_languages]
    lang_pairs_display = {
        language_mapping.get(src, src): [language_mapping.get(tgt, tgt) for tgt in tgts]
        for src, tgts in lang_pairs.items()
    }
    return source_languages_display, lang_pairs, lang_pairs_display

# 更新目标语言下拉框
def update_target_dropdown_with_mapping(src_lang_display):
    src_lang = reverse_language_mapping.get(src_lang_display, src_lang_display)
    if src_lang in lang_pairs:
        tgt_langs_display = [language_mapping.get(tgt, tgt) for tgt in lang_pairs[src_lang]]
        return gr.update(choices=tgt_langs_display, value=None)
    return gr.update(choices=[], value=None)

# 获取最新模型版本
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
        return langpair_folder
        #return None
    else:
        max_version_folder = max(file_list)[1]
        return os.path.join(langpair_folder, max_version_folder)
# 当用户切换源语言时，检查目标语言，并翻译文本框内容
# def handle_language_change(src_lang, tgt_lang, text, model, tokenizer):
#         if text and model and tokenizer:  # 如果文本框中有内容，立即翻译
#             return perform_translation(text, model, tokenizer)
#         return gr.update()
# 文本框延迟翻译逻辑 
def delayed_translation(text, model, tokenizer):
        time.sleep(0.5)  # 延迟 0.5 秒
        if model and tokenizer and text:
            return perform_translation(text, model, tokenizer)
        return gr.update() 
def perform_translation(text, model, tokenizer): 
    """执行翻译"""
    if model is None or tokenizer is None:
        return "请先选择语言对并加载模型！"  # 提示用户加载模型
    if not text:
        return ""  # 如果输入为空，返回空字符串
    try:
        logger.info(f"用户的输入是{text}")
        # 对输入文本进行编码
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        # 使用模型生成翻译
        outputs = model.generate(inputs["input_ids"])
        # 解码输出，去掉特殊符号
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"自研模型翻译结果是{res}")
        return res
    except Exception as e:
        return f"翻译出错：{e}"  # 返回错误信息
def load_model_and_tokenizer(model_path):
    model_path = get_max_version_folder(model_path)
    if model_path is None or not os.path.isdir(model_path):
        return None, None, "failure"
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    return model, tokenizer, "success"
def check_and_load_model(src_lang_display, tgt_lang_display, text, model, tokenizer):
    """检查语言对并加载对应的翻译模型，同时处理文本翻译"""
    src_lang = reverse_language_mapping.get(src_lang_display, src_lang_display)
    tgt_lang = reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)

    logger.info("=============================================")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"时间: {current_time}")

    if src_lang and tgt_lang:  # 确保源语言和目标语言不为空
        # 检查语言对是否可用，并获取模型路径
        status, model_path = delayed_language_check(src_lang, tgt_lang)
        if model_path:  
            # 加载模型和分词器
            logger.info(f"选择的源语言: {src_lang_display} ({src_lang}), 目标语言: {tgt_lang_display} ({tgt_lang})")
            logger.info(f"找到的模型路径: {model_path}")
            model, tokenizer, load_status = load_model_and_tokenizer(model_path)

            if load_status == "success":
                # 翻译文本框内容（如果有）
                if text:
                    translation = perform_translation(text, model, tokenizer)
                else:
                    translation = gr.update()
                
                # 返回加载的模型、分词器、输入框状态更新、翻译结果
                return model, tokenizer, gr.update(interactive=True, placeholder="请输入要翻译的文本..."), translation, model_path
            else:
                # 加载失败
                return None, None, gr.update(interactive=False, placeholder="模型加载失败，请检查配置。"), gr.update(),None
        else:
            # 如果模型不可用，禁用输入框
            return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。"), gr.update(),None
    
    # 如果语言对未指定，禁用输入框
    return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。"), gr.update(),None

# 延迟检查语言对可用性
def delayed_language_check(src_lang, tgt_lang):
    time.sleep(0.5)
    model_path, exists = config_data.get(src_lang, {}).get(tgt_lang, None), src_lang in config_data and tgt_lang in config_data[src_lang]
    if exists:
        return "success", model_path
    else:
        return "failure", None

# GPT-4o 翻译
def perform_gpt_translation(text, src_lang_display, tgt_lang_display):
    if not text:
        return ""
    src_lang = reverse_language_mapping.get(src_lang_display, src_lang_display)
    tgt_lang = reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
    client = openai.AzureOpenAI(
        azure_endpoint="https://tlsm-gpt4o-test2.openai.azure.com/",
        api_key="2dd9bb411f6741f6bebfddb016a3698f",
        api_version="2024-07-01-preview",
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个专业的翻译员，只输出目标语言的翻译结果"},
                {"role": "user", "content": f"源语言是 {src_lang}, 内容是：{text}，目标语言是{tgt_lang}，你需要先把内容全部转化小写再翻译"}
            ]
        )
        
        logger.info(f"GPT-4o翻译结果是{response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        return "Translation failed."

# Google 翻译
def perform_google_translation(text, src_lang_display, tgt_lang_display):
    if not text:
        return ""
    src_lang = reverse_language_mapping.get(src_lang_display, src_lang_display)
    tgt_lang = reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
    url = "https://translation.googleapis.com/language/translate/v2"
    api_key = "AIzaSyAzVTWGdfo16u9KLXIl0fObVefb0kPih_U"
    params = {
        "q": text,
        "target": tgt_lang,
        "key": api_key
    }
    if src_lang:
        params["source"] = src_lang
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        return result["data"]["translations"][0]["translatedText"]
    else:
        return f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}"

# 保存反馈
def save_badcase(comments, input_text, output_text, google_output_text, src_lang_display, tgt_lang_display, model_path):
    src_lang = reverse_language_mapping.get(src_lang_display, src_lang_display)
    tgt_lang = reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
    badcase = {
        "comments": comments,
        "input_text": input_text,
        "output_text": output_text,
        "google_output_text": google_output_text,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model_path":model_path
    }
    with open("badcase.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(badcase, ensure_ascii=False) + "\n")
    return "反馈已保存成功！"


with gr.Blocks() as demo:
    gr.HTML("""
    <div style="text-align: center; padding-top: 10px;">
    <div class="bili-avatar" style="margin-bottom: -10px;">
        <a href="https://cn.timekettle.co" target="_blank">
            <img style="width:120px;height:120px;border-radius:80px;max-width:120px;" 
                title="前往时空壶"
                src="https://26349372.s21i.faiusr.com/4/ABUIABAEGAAgmIf5gwYoluervAUwjBE4sRM.png">
        </a>
    </div>
    <h1 style="margin-top: 5px;">时空壶自研AI翻译助手</h1>
    </div>
    """)

    logger.add("log.txt", 
           rotation="10 MB",   # 文件达到 10 MB 时创建新文件
           retention=None,     # 设置为 None 表示永久保留
           encoding="utf-8",   # 确保支持中文
           level="INFO")    

    # 加载语言对映射
    source_languages_display, lang_pairs, lang_pairs_display = get_lang_pairs_with_mapping(config_data, language_mapping)

    with gr.Row():
        with gr.Column():
            src_lang_dropdown = gr.Dropdown(choices=source_languages_display, label="源语言", interactive=True)

            input_text = gr.Textbox(label="输入文本", lines=13, placeholder="请先选择源语言和目标语言。", interactive=False)
        
        with gr.Column():
            tgt_lang_dropdown = gr.Dropdown(choices=[], label="目标语言", interactive=True)
            output_text = gr.Textbox(label="模型翻译结果", lines=5, interactive=False)
            google_output_text = gr.Textbox(label="GPT-4o翻译结果", lines=5, interactive=False)
    
    with gr.Row():
        google_translate_button = gr.Button("使用GPT-4o翻译", elem_id="google_translate_button")

    with gr.Row():
        comments_textbox = gr.Textbox(label="欢迎您留下您的意见反馈", lines=5, placeholder="感谢使用！觉得翻译还行吗？不妨在下面写点反馈，您的吐槽或表扬都会让我们的产品更优秀哦！🎉", elem_id="comments_textbox")
    with gr.Row():   
        save_badcase_button = gr.Button("感谢您的反馈，写完请点这里提交！", elem_id="save_badcase_button")
    
    model_var = gr.State()
    tokenizer_var = gr.State()
    model_path_var = gr.State()

    src_lang_dropdown.change(fn=update_target_dropdown_with_mapping, 
                            inputs=src_lang_dropdown, 
                            outputs=tgt_lang_dropdown)


    # tgt_lang_dropdown.change(
    #     fn=check_and_load_model,
    #     inputs=[src_lang_dropdown, tgt_lang_dropdown],
    #     outputs=[model_var, tokenizer_var, input_text]
    # )

    tgt_lang_dropdown.change(
    fn=check_and_load_model,
    inputs=[src_lang_dropdown, tgt_lang_dropdown, input_text, model_var, tokenizer_var],
    outputs=[model_var, tokenizer_var, input_text, output_text,model_path_var]
    )


    

    input_text.change(fn=perform_translation, 
                    inputs=[input_text, model_var, tokenizer_var], 
                    outputs=output_text)

    google_translate_button.click(fn=perform_gpt_translation, 
                    inputs=[input_text, src_lang_dropdown, tgt_lang_dropdown], 
                    outputs=google_output_text)


    save_badcase_button.click(
        fn=save_badcase,
        inputs=[comments_textbox, input_text, output_text, google_output_text, src_lang_dropdown, tgt_lang_dropdown,model_path_var],
        outputs=gr.Textbox(label="提交状态", value="等待用户输入反馈评价！"),
    )

    gr.HTML("""
        <div style="text-align:center">           
            <br/>
            <br/>
            仅供学习交流，不可用于或非法用途
            <br/>
        </div>
    """)

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

#save_badcase_button {
    width: 10px;  /* 根据内容自动调整宽度 */
    margin: 1px auto;
    padding: 10px; /* 减小内边距 */
    font-size: 16px;
    font-weight: bold;
    color: linear-gradient(135deg, #ffa07a, #ff7f50);
    background: linear-gradient(135deg, #ffa07a, #ff7f50);
    border: 2px solid linear-gradient(135deg, #ffa07a, #ff7f50);
    border-radius: 8px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

#save_badcase_button:hover {
    background: linear-gradient(135deg, #ff7f50, #ffa07a);
    color: linear-gradient(135deg, #ffa07a, #ff7f50);
}

#save_badcase_button:active {
    background: linear-gradient(135deg, #ffa07a, #ff7f50);
    color: linear-gradient(135deg, #ffa07a, #ff7f50);
}

#comments_textbox {
    width: 800px ; 
    margin: 0 auto; /* 使其居中 */
    font-size: 16px; /* 调整字体大小 */
    padding: 10px; /* 增加内边距 */
    box-sizing: border-box; /* 确保宽度包含内边距 */
}



"""
demo.launch(server_name="0.0.0.0",server_port=7983)
