from transformers import MarianMTModel, MarianTokenizer
import gradio as gr
import os
import json
import time
import requests
import openai
from loguru import logger
from datetime import datetime
from transformers import MarianTokenizer
import torch

class TranslationApp:
    def __init__(self, config_file, mapping_file, multilanguage_config_file = "config/multilang_config.json"):
        self.config_file = config_file
        self.multilanguage_config_file = multilanguage_config_file
        self.mapping_file = mapping_file
        
        self.config_data = self.load_config()
        self.multilang_config = self.load_multilang_config()
        self.multi_language_choices = list(self.multilang_config.keys())
        self.language_mapping, self.reverse_language_mapping = self.load_language_mapping()
        self.source_languages_display, self.lang_pairs, self.lang_pairs_display = self.get_lang_pairs_with_mapping()
        
        logger.add("output/user_log.txt", rotation="10 MB", retention=None, encoding="utf-8", level="INFO")
        self.tokenizer = None
        self.language_ids = None
        self.model_path = None
        


    
    
    def load_config(self):
        """加载 config.json 配置文件"""
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def load_multilang_config(self):
        """加载多语种模型的配置文件"""
        if not os.path.exists(self.multilanguage_config_file):
            raise FileNotFoundError(f"配置文件 {self.multilanguage_config_file} 未找到！")
        with open(self.multilanguage_config_file, 'r', encoding='utf-8') as f:
            return json.load(f)


    def load_language_mapping(self):
        """加载语言映射文件"""
        language_mapping = {}
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                mapping = json.loads(line.strip())
                language_mapping.update(mapping)
        
        """创建反向映射"""
        reverse_mapping = {v: k for k, v in language_mapping.items()}
        return language_mapping, reverse_mapping

    def load_multilanguage_mapping(self):
        """加载语言标识符到中文名称的映射"""
        if not os.path.exists(self.mapping_file):
            raise FileNotFoundError(f"语言映射文件 {self.mapping_file} 未找到！")
        language_mapping = {}
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                mapping = json.loads(line.strip())
                language_mapping.update(mapping)
        reverse_mapping = {v: k for k, v in language_mapping.items()}
        return language_mapping, reverse_mapping

    def get_lang_pairs_with_mapping(self):
        """获取语言对映射，显示中文名称"""
        source_languages = list(self.config_data.keys())
        lang_pairs = {src: list(targets.keys()) for src, targets in self.config_data.items()}


        source_languages_display = [self.language_mapping.get(src, src) for src in source_languages]
        lang_pairs_display = {
            self.language_mapping.get(src, src): [self.language_mapping.get(tgt, tgt) for tgt in tgts]
            for src, tgts in lang_pairs.items()
        }
        return source_languages_display, lang_pairs, lang_pairs_display

    def update_target_dropdown_with_mapping(self, src_lang_display):
        """更新目标语言下拉框"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        if src_lang in self.lang_pairs:
            tgt_langs_display = [self.language_mapping.get(tgt, tgt) for tgt in self.lang_pairs[src_lang]]
            return gr.update(choices=tgt_langs_display, value=None)
        return gr.update(choices=[], value=None)

    def get_max_version_folder(self, langpair_folder):
        """获取最新模型版本"""
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

    def delayed_translation(self, text, model, tokenizer):
        """延迟翻译逻辑"""
        time.sleep(0.5)  # 延迟 0.5 秒
        if model and tokenizer and text:
            return self.perform_translation(text, model, tokenizer)
        return gr.update()
    
    def mask_language_token(self, input_ids: torch.Tensor, unk_token: int) -> torch.Tensor:
        """
        针对多语言模型，随机掩盖源语种、目标语种
        """
        aug_type = random.choices(
            ["mask_src", "mask_tgt", "mask_both", "no_mask"],
            [0.15, 0.1, 0.05, 0.7]
        )[0]

        if aug_type == "mask_src":
            input_ids[0] = unk_token
        elif aug_type == "mask_tgt":
            input_ids[1] = unk_token
        elif aug_type == "mask_both":
            input_ids[0] = unk_token
            input_ids[1] = unk_token
        elif aug_type == "no_mask":
            pass

        return input_ids

    def add_language_ids(
        self, 
        inputs, 
        src_lang: str, 
        tgt_lang: str,
        ) -> torch.Tensor:

        
        #inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)

        self.language_ids = {
                language: _id
                for language, _id in self.tokenizer.encoder.items()
                if language.startswith(">>") and language.endswith("<<")
        }

        # 值是这些语言标识符对应的整数索引
        src_lang_id = torch.tensor([self.language_ids[f">>{src_lang}<<"]])
        tgt_lang_id = torch.tensor([self.language_ids[f">>{tgt_lang}<<"]])
        
        # 将源语言标识符（src_lang_id）、目标语言标识符（tgt_lang_id）以及输入的 input_ids 拼接在一起，形成一个新的输入序列，并传递给模型以明确指定翻译方向。
        # e.g :
        # src_lang_id = torch.tensor([1]) # 英语标识符 
        # tgt_lang_id = torch.tensor([2]) # 法语标识符
        # inputs["input_ids"][:-2] = torch.tensor([101, 7592, 1010, 2129])
        # tensor([1, 2, 101, 7592, 1010, 2129])
        inputs["input_ids"] = torch.cat(
            (src_lang_id, tgt_lang_id, inputs["input_ids"]), dim=-1
        ).unsqueeze(0)

        # 指示哪些标记是有效的（即非填充的），用于模型在计算时忽略填充部分。1 表示实际的，0表示padding
        inputs["attention_mask"] = torch.cat(
            (torch.ones(2), inputs["attention_mask"]), dim=-1
        ).unsqueeze(0)


        inputs["src_lang"] = src_lang
        inputs["tgt_lang"] = tgt_lang

        inputs 
        return inputs
    
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
        """加载模型和分词器"""
        model_path = self.get_max_version_folder(model_path)
        if model_path is None or not os.path.isdir(model_path):
            return None, None, "failure"
        model = MarianMTModel.from_pretrained(model_path)
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.tokenizer = tokenizer
        return model, tokenizer, "success"

    def check_and_load_model(self, src_lang_display, tgt_lang_display, text, model, tokenizer):
        """检查语言对并加载模型"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)

        logger.info("=============================================")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"时间: {current_time}")

        if src_lang and tgt_lang:
            status, model_path = self.delayed_language_check(src_lang, tgt_lang)
            if model_path:
                logger.info(f"选择的源语言: {src_lang_display} ({src_lang}), 目标语言: {tgt_lang_display} ({tgt_lang})")
                logger.info(f"找到的模型路径: {model_path}")
                model, tokenizer, load_status = self.load_model_and_tokenizer(model_path)

                if load_status == "success":
                    if text:
                        translation = self.perform_translation(text, model, tokenizer)
                    else:
                        translation = gr.update()
                    return model, tokenizer, gr.update(interactive=True, placeholder="请输入要翻译的文本..."), translation, model_path
                else:
                    return None, None, gr.update(interactive=False, placeholder="模型加载失败，请检查配置。"), gr.update(), None
            else:
                return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。"), gr.update(), None
        return None, None, gr.update(interactive=False, placeholder="请先选择源语言和目标语言。"), gr.update(), None

    def delayed_language_check(self, src_lang, tgt_lang):
        """延迟检查语言对可用性"""
        time.sleep(0.5)
        model_path, exists = self.config_data.get(src_lang, {}).get(tgt_lang, None), src_lang in self.config_data and tgt_lang in self.config_data[src_lang]
        if exists:
            return "success", model_path
        else:
            return "failure", None

    def perform_gpt_translation(self, text, src_lang_display, tgt_lang_display):
        """使用GPT-4o翻译"""
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

    def save_badcase(self, comments, input_text, output_text, google_output_text, src_lang_display, tgt_lang_display, model_path):
        """保存反馈"""
        src_lang = self.reverse_language_mapping.get(src_lang_display, src_lang_display)
        tgt_lang = self.reverse_language_mapping.get(tgt_lang_display, tgt_lang_display)
        badcase = {
            "comments": comments,
            "input_text": input_text,
            "output_text": output_text,
            "google_output_text": google_output_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model_path": model_path
        }
        with open("output/badcase.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(badcase, ensure_ascii=False, indent=2) + "\n")
        return "反馈已保存成功！"

    def launch_app(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    src_lang_dropdown = gr.Dropdown(choices=self.source_languages_display, label="源语言", interactive=True)
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

            src_lang_dropdown.change(fn=self.update_target_dropdown_with_mapping, 
                                    inputs=src_lang_dropdown, 
                                    outputs=tgt_lang_dropdown)

            tgt_lang_dropdown.change(fn=self.check_and_load_model,
                                    inputs=[src_lang_dropdown, tgt_lang_dropdown, input_text, model_var, tokenizer_var],
                                    outputs=[model_var, tokenizer_var, input_text, output_text, model_path_var])

            input_text.change(fn=self.perform_translation, 
                            inputs=[input_text, model_var, tokenizer_var], 
                            outputs=output_text)

            google_translate_button.click(fn=self.perform_gpt_translation, 
                                inputs=[input_text, src_lang_dropdown, tgt_lang_dropdown], 
                                outputs=google_output_text)

            save_badcase_button.click(fn=self.save_badcase,
                                    inputs=[comments_textbox, input_text, output_text, google_output_text, src_lang_dropdown, tgt_lang_dropdown, model_path_var],
                                    outputs=gr.Textbox(label="提交状态", value="等待用户输入反馈！"))

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
        #demo.launch(server_name="0.0.0.0", server_port=7989)
    
    def get_multilang_model(self, model_name):
        """
        根据用户选择的模型名加载对应模型
        """
        self.model_path = self.multilang_config.get(model_name)
        if self.model_path is None or not os.path.isdir(self.model_path):
            logger.warning(f"未找到模型或模型路径无效: {model_name}")
            return gr.update(interactive=False, placeholder="模型加载失败，请检查路径。")

        try:
            # 加载模型和 tokenizer
            model = MarianMTModel.from_pretrained(self.model_path)
            tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.tokenizer = tokenizer  # 更新类中的 tokenizer
            self.model = model  # 更新类中的 model
            logger.info(f"成功加载模型: {model_name}, 路径: {self.model_path}")

            return gr.update(interactive=False, placeholder="请选择源语言和目标语言"),self.model_path
        except Exception as e:
            logger.error(f"模型加载错误: {e}")
            return gr.update(interactive=False, placeholder="模型加载失败，请检查配置。"),None




    def update_source_languages(self, model_name):
        """
        加载模型时解析 train_info.json，并更新源语言列表
        """
        model_path = self.multilang_config.get(model_name)
        if not model_path:
            return gr.update(choices=[], interactive=False), gr.update(choices=[], interactive=False)

        train_info_path = os.path.join(model_path, "train_info.json")
        if not os.path.isfile(train_info_path):
            return gr.update(choices=[], interactive=False), gr.update(choices=[], interactive=False)

        with open(train_info_path, 'r', encoding='utf-8') as f:
            train_info = json.load(f)

        lang_pairs = train_info.get("lang_pairs", {})
        source_languages = list(lang_pairs.keys())

        # 加载语言映射
        language_mapping, reverse_mapping = self.load_multilanguage_mapping()

        # 将源语言标识符转换为中文
        source_languages_chinese = [language_mapping.get(lang, lang) for lang in source_languages]
        
        # 更新源语言下拉框
        return gr.update(choices=source_languages_chinese, interactive=True), gr.update(choices=[], interactive=False)

    
    def update_target_languages(self, source_lang, model_name):
        """
        根据源语言动态更新目标语言选项
        """
        model_path = self.multilang_config.get(model_name)
        if not model_path:
            return gr.update(choices=[], interactive=False)

        train_info_path = os.path.join(model_path, "train_info.json")
        if not os.path.isfile(train_info_path):
            return gr.update(choices=[], interactive=False)

        with open(train_info_path, 'r', encoding='utf-8') as f:
            train_info = json.load(f)

        lang_pairs = train_info.get("lang_pairs", {})
        
        # 确保从源语言选择中获取正确的键值
        # 如果 source_lang 是中文，可能需要将其转换为对应的英文或源语言标识符
        language_mapping, reverse_mapping = self.load_multilanguage_mapping()
        source_lang_key = reverse_mapping.get(source_lang, source_lang)  # 转换为源语言标识符

        target_languages = lang_pairs.get(source_lang_key, [])

        # 将目标语言标识符转换为中文
        target_languages_chinese = [language_mapping.get(lang, lang) for lang in target_languages]

        return gr.update(choices=target_languages_chinese, interactive=True)


    # 控制输入框启用状态
    def control_input_activation(self, source_lang, target_lang,model_name):
        """
        确保当用户选择了源语言和目标语言时才启用输入框
        """
        if source_lang and target_lang and model_name:
            return gr.update(interactive=True,placeholder="请输入")
        else:
            return gr.update(interactive=False)


    def perform_multilang_translation(self, text, source_lang_chinese, target_lang_chinese):
        """
        执行多语种翻译
        """
        if not text or not self.model or not self.tokenizer:
            return "请先选择多语种模型并设置语言对！"

        language_mapping, reverse_mapping = self.load_multilanguage_mapping()
        source_lang = reverse_mapping.get(source_lang_chinese, source_lang_chinese)
        target_lang = reverse_mapping.get(target_lang_chinese, target_lang_chinese)

            
        #formatted_text = f">>{target_lang}<< {text}"
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs = self.add_language_ids(inputs,source_lang, target_lang)
        outputs = self.model.generate(inputs["input_ids"])
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation



    
    
    def render_layout(self):
        with gr.Blocks() as demo:
            gr.HTML("""
            <div style="display: flex; align-items: center; padding-top: 10px;">
                <a href="https://cn.timekettle.co" target="_blank" style="margin-right: 10px;">
                    <img style="width: 120px; height: 120px; border-radius: 80px; max-width: 120px;" 
                         title="前往时空壶"
                         src="https://26349372.s21i.faiusr.com/4/ABUIABAEGAAgmIf5gwYoluervAUwjBE4sRM.png">
                </a>
                <div style="text-align: left;">
                    <h1 style="margin: 0; font-size: 32px;">时空壶翻译 Demo</h1>
                    <h2 style="margin: 0; font-size: 16px; color: gray; text-align: left;">基于Marian机器翻译模型，支持GPT-4o翻译对照</h2>
                </div>
            </div>
            """)
            with gr.Column():
                # 用户选择单语种或多语种翻译
                mode_dropdown = gr.Dropdown(
                    choices=["单语种翻译", "多语种翻译"],
                    label="翻译模式",
                    value="单语种翻译",
                    interactive=True,
                )

            # 单语种翻译布局
            with gr.Column(visible=True) as single_language_row:
                self.launch_app()

            # 多语种翻译布局
            with gr.Column(visible=False) as multi_language_row:
                model_status_output = gr.State()
                model_path_var = gr.State()

                # 多语种模型下拉框
                multi_language_model_dropdown = gr.Dropdown(
                    choices=self.multi_language_choices,
                    label="选择多语种模型",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column():
                        # 源语言下拉框
                        source_lang_dropdown = gr.Dropdown(
                            choices=[],  # 动态加载
                            label="选择源语言",
                            interactive=False,
                        )
                        multi_input_text = gr.Textbox(
                                label="输入文本",
                                lines=13,
                                placeholder="请选择多语种模型",
                                interactive=False,
                        )
                    with gr.Column():
                        # 目标语言下拉框
                        target_lang_dropdown = gr.Dropdown(
                            choices=[],  # 动态加载
                            label="选择目标语言",
                            interactive=False,
                        )

                        # 输出文本框
                        multi_output_text = gr.Textbox(
                            label="翻译结果", lines=5, interactive=False
                        )

                        # GPT-4o 翻译结果文本框
                        multi_google_output_text = gr.Textbox(
                            label="GPT-4o翻译结果", lines=5, interactive=False
                        )

                with gr.Row():
                    google_translate_button = gr.Button("使用GPT-4o翻译", elem_id="google_translate_button")

                with gr.Row():
                    comments_textbox = gr.Textbox(label="欢迎您留下您的意见反馈", lines=5, placeholder="感谢使用！觉得翻译还行吗？不妨在下面写点反馈，您的吐槽或表扬都会让我们的产品更优秀哦！🎉", elem_id="comments_textbox")
                with gr.Row():   
                    save_badcase_button = gr.Button("感谢您的反馈，写完请点这里提交！", elem_id="save_badcase_button")

                
                # 加载多语种模型
                multi_language_model_dropdown.change(
                    fn=self.get_multilang_model,  # 加载模型
                    inputs=[multi_language_model_dropdown],
                    outputs=[multi_input_text, model_path_var],
                )
                
                
                # 动态更新源语言
                multi_language_model_dropdown.change(
                    fn=self.update_source_languages,  # 加载模型并更新源语言列表
                    inputs=[multi_language_model_dropdown],
                    outputs=[source_lang_dropdown, target_lang_dropdown],
                )

                # 动态目标语言
                source_lang_dropdown.change(
                    fn=self.update_target_languages,  # 根据源语言更新目标语言
                    inputs=[source_lang_dropdown, multi_language_model_dropdown],
                    outputs=[target_lang_dropdown],
                )

                
                source_lang_dropdown.change(
                    fn=self.control_input_activation,  # 确保源语言选择后，输入框可以交互
                    inputs=[source_lang_dropdown, target_lang_dropdown, multi_language_model_dropdown],
                    outputs=[multi_input_text]
                )

                target_lang_dropdown.change(
                    fn=self.control_input_activation,  # 确保目标语言选择后，输入框可以交互
                    inputs=[source_lang_dropdown, target_lang_dropdown, multi_language_model_dropdown],
                    outputs=[multi_input_text]
                )

                # 设置输入框的翻译逻辑
                multi_input_text.change(
                    fn=self.perform_multilang_translation,
                    inputs=[multi_input_text, source_lang_dropdown, target_lang_dropdown],
                    outputs=multi_output_text,
                )

                google_translate_button.click(fn=self.perform_gpt_translation, 
                                inputs=[multi_input_text, source_lang_dropdown, target_lang_dropdown], 
                                outputs=multi_google_output_text)

                

                save_badcase_button.click(fn=self.save_badcase,
                                        inputs=[comments_textbox, multi_input_text, multi_output_text, multi_google_output_text, source_lang_dropdown, target_lang_dropdown, model_path_var],
                                        outputs=gr.Textbox(label="提交状态", value="等待用户输入反馈！"))
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


            # 动态切换布局
            def toggle_translation_mode(mode):
                if mode == "单语种翻译":
                    return gr.update(visible=True), gr.update(visible=False)
                elif mode == "多语种翻译":
                    return gr.update(visible=False), gr.update(visible=True)

            mode_dropdown.change(
                fn=toggle_translation_mode,
                inputs=[mode_dropdown],
                outputs=[single_language_row, multi_language_row],
            )

        return demo


if __name__ == "__main__":
    app = TranslationApp("config/user_config.json", "config/mapping.jsonl")
    demo = app.render_layout()
    demo.launch(server_name="0.0.0.0", server_port=7856)
