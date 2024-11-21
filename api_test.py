import openai

def gpt4_translate(text, target_lang, api_key):
    # 设置 OpenAI API 密钥

    client = openai.AzureOpenAI(
        azure_endpoint="https://tlsm-gpt4o-test2.openai.azure.com/",
        api_key=api_key,
        api_version="2024-07-01-preview",
    )
    
    # 调用 OpenAI GPT-4 翻译
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator. Please translate the following text."},
                {"role": "user", "content": f"Translate the following text into {target_lang}: {text}"}
            ]
        )
        # 返回翻译结果
        return response.choices[0].message.content
    except Exception as e:
        return "Translation failed."

# 示例调用
api_key = "2dd9bb411f6741f6bebfddb016a3698f"  # 替换为你的 OpenAI API 密钥
text = "Hello, how are you?"
target_lang = "Chinese"  # 目标语言（可以用英文名称，如 Chinese、French 等）
translated_text = gpt4_translate(text, target_lang, api_key)
print(translated_text)
