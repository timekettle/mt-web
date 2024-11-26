from transformers import MarianMTModel, MarianTokenizer
import gradio as gr
import os
import json
import time
import requests
import openai
from loguru import logger
from datetime import datetime
from typing import Dict,Tuple,List

class TranslationApp:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_data = self.load_config()
        self.language_mapping = self.get_language_mapping()
        self.reverse_language_mapping = self.get_reverse_language_mapping()
        self.source_languages_display, self.lang_pairs_display = self.get_lang_pairs_with_mapping()


    def load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config["languages"] 

    def get_language_mapping(self):
        """获取语言和其映射名称"""
        return {lang_key: lang_data["name"] for lang_key, lang_data in self.config_data.items()}

    def get_reverse_language_mapping(self):
        """获取反向映射（显示名称到语言缩写）"""
        reverse_mapping = {}
        for lang_id, lang_data in self.config_data.items():
            reverse_mapping[lang_data["name"]] = lang_id
            for target_id, target_data in lang_data.get("targets", {}).items():
                reverse_mapping[target_data["name"]] = target_id
        return reverse_mapping

    def get_lang_pairs_with_mapping(self):
        """获取语言对映射"""
        source_languages_display = list(self.language_mapping.values())
        lang_pairs_display = {
            self.language_mapping[src_lang]: [
                self.config_data[src_lang]["targets"][tgt]["name"] 
                for tgt in self.config_data[src_lang]["targets"]
            ]
            for src_lang in self.config_data
        }
        return source_languages_display, lang_pairs_display

    def get_language_key(self, lang_display: str) -> str:
        """通过显示名称获取语言的缩写"""
        for lang_key, lang_data in self.config_data.items():
            if lang_data["name"] == lang_display:
                return lang_key
        return lang_display

    def update_target_dropdown_with_mapping(self, src_lang_display):
        """更新目标语言下拉框"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        if src_lang in self.config_data:
            tgt_langs_display = [
                self.config_data[src_lang]["targets"][tgt]["name"] 
                for tgt in self.config_data[src_lang]["targets"]
            ]
            return gr.update(choices=tgt_langs_display, value=None)
        return gr.update(choices=[], value=None)

    def update_model_dropdown(self, src_lang_display, tgt_lang_display):
        """更新模型路径下拉框"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)  # 修复目标语言映射问题
        
        logger.info(f"源语言映射: {src_lang_display} -> {src_lang}, 目标语言映射: {tgt_lang_display} -> {tgt_lang}")

        # 检查目标语言是否有效
        if not tgt_lang_display or tgt_lang is None:
            logger.warning(f"目标语言未选择，无法更新模型路径。源语言映射: {src_lang_display} -> {src_lang}")
            return gr.update(choices=[], value=None)

        if src_lang in self.config_data and tgt_lang in self.config_data[src_lang]["targets"]:
            model_paths = self.config_data[src_lang]["targets"][tgt_lang].get("model_paths", [])
            logger.info(f"找到的模型路径: {model_paths}")
            return gr.update(choices=model_paths, value=None)
        else:
            logger.warning(f"语言对 ({src_lang}, {tgt_lang}) 无法找到模型路径")
        
        return gr.update(choices=[], value=None)




    def get_max_version_folder(self, langpair_folder):
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
        else:
            max_version_folder = max(file_list)[1]
            return os.path.join(langpair_folder, max_version_folder)

    def perform_translation(self, text, model, tokenizer):
        """执行翻译"""
        if model is None or tokenizer is None:
            return "请先选择语言对并加载模型！"
        if not text:
            return ""
        try:
            logger.info(f"用户的输入是{text}")
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(inputs["input_ids"])
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"自研模型翻译结果是{res}")
            return res
        except Exception as e:
            return f"翻译出错：{e}"

    def load_model_and_tokenizer(self, model_path):
        model_path = self.get_max_version_folder(model_path)
        if model_path is None or not os.path.isdir(model_path):
            return None, None, "failure"
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        return model, tokenizer, "success"
    


    def check_and_load_model(self, src_lang_display, tgt_lang_display, model_path, text, model, tokenizer):
        """检查语言对、加载对应的翻译模型，并处理文本翻译"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)

        logger.info("=============================================")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"时间: {current_time}")

        if src_lang and tgt_lang and model_path:  # 确保源语言、目标语言和模型路径都已选择
            logger.info(f"源语言: {src_lang_display} ({src_lang}), 目标语言: {tgt_lang_display} ({tgt_lang}), 模型路径: {model_path}")
            # 加载模型和分词器
            model, tokenizer, load_status = self.load_model_and_tokenizer(model_path)

            if load_status == "success":
                logger.info("模型加载成功")
                if text:  # 如果有输入文本，执行翻译
                    translation = self.perform_translation(text, model, tokenizer)
                else:
                    translation = gr.update()
                return model, tokenizer, gr.update(interactive=True, placeholder="模型已加载，输入文本开始翻译。"), translation,model_path
            else:
                # 模型加载失败
                return None, None, gr.update(interactive=False, placeholder="模型加载失败，请检查路径。"), gr.update(), None
        
        # 如果路径未选择，禁用输入框
        return None, None, gr.update(interactive=False, placeholder="请先选择模型路径。"), gr.update(), None


    def perform_gpt_translation(self, text, src_lang_display, tgt_lang_display):
        if not text:
            return ""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
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
    
    def delayed_language_check(self, src_lang, tgt_lang):
        time.sleep(0.5)
        model_path, exists = self.config_data.get(src_lang, {}).get(tgt_lang, None), src_lang in self.config_data and tgt_lang in self.config_data[src_lang]
        if exists:
            return "success", model_path
        else:
            return "failure", None

    def save_badcase(self,comments, input_text, output_text, google_output_text, src_lang_display, tgt_lang_display, model_path):
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
        badcase = {
            "comments": comments,
            "input_text": input_text,
            "output_text": output_text,
            "gpt_output_text": google_output_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_path":model_path
        }
        with open("output/badcase.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(badcase, ensure_ascii=False,indent=2) + "\n")
        return "反馈已保存成功！"



if __name__ == "__main__":
    app = TranslationApp("config/deve_config.json")

    with gr.Blocks() as demo:
        gr.HTML("""
        <div style="display: flex; align-items: center; padding-top: 10px;">
        <!-- 图片部分 -->
        <a href="https://cn.timekettle.co" target="_blank" style="margin-right: 10px;">
            <img style="width: 120px; height: 120px; border-radius: 80px; max-width: 120px;" 
                title="前往时空壶"
                src="https://26349372.s21i.faiusr.com/4/ABUIABAEGAAgmIf5gwYoluervAUwjBE4sRM.png">
        </a>
        <!-- 标题部分 -->
        <div style="text-align: left;">
            <h1 style="margin: 0; font-size: 32px;">时空壶翻译(Timekettle Translator) Demo</h1>
            <h2 style="margin: 0; font-size: 16px; color: gray; text-align: left;">基于Marian机器翻译模型，支持GPT-4o翻译对照，开发者使用版</h2>
        </div>
        </div>
        """)

        logger.add("output/deve_log.txt", rotation="10 MB", retention=None, encoding="utf-8", level="INFO")

        with gr.Row():
            with gr.Column():
                src_lang_dropdown = gr.Dropdown(choices=app.source_languages_display, label="源语言", interactive=True)
                input_text = gr.Textbox(label="输入文本", lines=13, placeholder="请先选择源语言和目标语言。", interactive=False)
            with gr.Column():
                # 目标语言选择
                tgt_lang_dropdown = gr.Dropdown(choices=[], label="目标语言", interactive=True)
                # 模型选择
                model_dropdown = gr.Dropdown(choices=[], label="具体模型", interactive=True)
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

        # 源语言选择更新目标语言
        src_lang_dropdown.change(
            fn=app.update_target_dropdown_with_mapping,
            inputs=[src_lang_dropdown],
            outputs=[tgt_lang_dropdown]
        )

        # 目标语言选择更新模型路径
        tgt_lang_dropdown.change(
            fn=app.update_model_dropdown,
            inputs=[src_lang_dropdown, tgt_lang_dropdown],
            outputs=[model_dropdown]
        )

        # 加载模型
        model_dropdown.change(
        fn=app.check_and_load_model,
        inputs=[src_lang_dropdown, tgt_lang_dropdown, model_dropdown, input_text, model_var, tokenizer_var],
        outputs=[model_var, tokenizer_var, input_text, output_text,model_path_var]
        )
        
        # 用户输入并翻译
        input_text.change(fn=app.perform_translation, 
                        inputs=[input_text, model_var, tokenizer_var], 
                        outputs=output_text)

        google_translate_button.click(fn=app.perform_gpt_translation, 
                        inputs=[input_text, src_lang_dropdown, tgt_lang_dropdown], 
                        outputs=google_output_text)


        save_badcase_button.click(
            fn=app.save_badcase,
            inputs=[comments_textbox, input_text, output_text, google_output_text, src_lang_dropdown, tgt_lang_dropdown,model_path_var],
            outputs=gr.Textbox(label="提交状态", value="等待用户输入反馈评价！"),
        )
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
        

    demo.launch(server_name="0.0.0.0", server_port=7961)
