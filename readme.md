# 机器翻译界面项目
本项目是一个基于 MarianMT 和 GPT-4o 的机器翻译应用，提供了用户友好的界面，用于文本翻译和多语言支持。项目还集成了日志记录、错误反馈和分布式模型加载等功能。

## 功能概述

- **多语言翻译支持**：
  - 基于 MarianMT 模型的本地翻译。
  - 支持 GPT-4o 翻译（Azure OpenAI 服务）。
  - 集成 Google Translate API 作为备选。

- **用户交互界面**：
  - 提供源语言和目标语言选择。
  - 显示翻译结果和对比。

- **日志记录**：
  - 自动记录用户操作、时间和翻译详情。

- **用户反馈**：
  - 提供反馈提交功能，保存用户建议和意见。


## 项目目录结构
├── config.json          
├── mapping.jsonl        
├── log.txt              
├── badcase.jsonl        
├── infer.py              

## 启动程序
- 启动应用：
  python infer.py
- 打开浏览器，访问：
  http://localhost:7999

## License

本项目采用 MIT 许可证授权，详细信息请参阅 LICENSE 文件。

## Acknowledgements
本项目基于 Gradio 框架构建，并在 MIT 许可证 下发布。参考了项目 vits-web 的设计与实现。[https://github.com/timekettle/vits-web]