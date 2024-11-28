import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from loguru import logger

class Multilang:
    def __init__(self, multilanguage_config_file="config/multilang_config.json"):
        self.multilanguage_config_file = multilanguage_config_file
        self.multilang_config = self.load_multilang_config()
        self.language_ids = None
        self.tokenizer = None
        self.model = None

    def load_multilang_config(self):
        """加载多语种模型的配置文件"""
        if not os.path.exists(self.multilanguage_config_file):
            raise FileNotFoundError(f"配置文件 {self.multilanguage_config_file} 未找到！")
        with open(self.multilanguage_config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_multilang_model(self, model_name):
        """
        根据用户选择的模型名加载对应模型
        """
        model_path = self.multilang_config.get(model_name)
        try:
            model = MarianMTModel.from_pretrained(model_path)
            tokenizer = MarianTokenizer.from_pretrained(model_path)

            train_info_path = os.path.join(model_path, "train_info.json")
            if not os.path.exists(train_info_path):
                return None, None, f"模型配置文件 train_info.json 未找到在路径: {model_path}"

            with open(train_info_path, 'r', encoding='utf-8') as f:
                train_info = json.load(f)

            lang_pairs = train_info.get("lang_pairs", {})
            if not lang_pairs:
                return None, None, "没有语言对信息，请检查模型配置文件。"

            self.model = model
            self.tokenizer = tokenizer
            return model, tokenizer, lang_pairs

        except Exception as e:
            print(f"模型加载错误: {e}")
            return None, None, f"模型加载失败，错误信息: {e}"

    def add_language_ids(self, inputs, src_lang, tgt_lang):
        """
        给输入添加语言标识符，以明确翻译方向。
        """
        # 获取语言标识符对应的ID
        self.language_ids = {
            language: _id
            for language, _id in self.tokenizer.encoder.items()
            if language.startswith(">>") and language.endswith("<<")
        }

        # 获取源语言和目标语言的标识符ID
        src_lang_id = torch.tensor([self.language_ids[f">>{src_lang}<<"]])
        tgt_lang_id = torch.tensor([self.language_ids[f">>{tgt_lang}<<"]])

        # 拼接源语言标识符、目标语言标识符和输入的文本序列
        inputs["input_ids"] = torch.cat((src_lang_id, tgt_lang_id, inputs["input_ids"]), dim=-1).unsqueeze(0)
        inputs["attention_mask"] = torch.cat((torch.ones(2), inputs["attention_mask"]), dim=-1).unsqueeze(0)
        inputs["src_lang"] = src_lang
        inputs["tgt_lang"] = tgt_lang

        return inputs

    def perform_multilang_translation(self, text, source_lang, target_lang):
        """
        执行多语种翻译
        """
        if not text or not self.model or not self.tokenizer:
            return "请先选择多语种模型并设置语言对！"

        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs = self.add_language_ids(inputs, source_lang, target_lang)
        
        # 执行翻译
        outputs = self.model.generate(inputs["input_ids"])
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation




def main():
    
    

    multilang = Multilang()
    

    print("请选择一个多语种模型：")
    for idx, model_name in enumerate(multilang.multilang_config.keys(), 1):
        print(f"{idx}. {model_name}")
    
    # 选择模型
    model_choice = int(input("请输入模型编号：")) - 1
    model_names = list(multilang.multilang_config.keys())
    selected_model = model_names[model_choice]
    
    # 加载模型及语言对
    model, tokenizer, lang_pairs = multilang.get_multilang_model(selected_model)
    if model is None:
        print("模型加载失败，程序退出。")
        return

    print(f"加载成功！当前模型: {selected_model}")
    
    # 显示源语言和目标语言选项
    print("\n可用的源语言：")
    source_languages = list(lang_pairs.keys())
    for idx, lang in enumerate(source_languages, 1):
        print(f"{idx}. {lang}")
    
    source_choice = int(input("请输入源语言编号：")) - 1
    source_lang = source_languages[source_choice]
    
    print(f"选择的源语言: {source_lang}")
    
    print("\n可用的目标语言：")
    target_languages = lang_pairs.get(source_lang, [])
    for idx, lang in enumerate(target_languages, 1):
        print(f"{idx}. {lang}")
    
    target_choice = int(input("请输入目标语言编号：")) - 1
    target_lang = target_languages[target_choice]
    
    print(f"选择的目标语言: {target_lang}")
    
    # 输入待翻译文本
    text_to_translate = input("请输入要翻译的文本：")
    
    # 执行翻译
    translation = multilang.perform_multilang_translation(text_to_translate, source_lang, target_lang)
    print(f"\n翻译结果：{translation}")

if __name__ == "__main__":
    main()