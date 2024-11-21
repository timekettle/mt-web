import gradio as gr
from tts import Synthesize


def info():
    info_tab = gr.Markdown(
        '<div align="left">'
        '</br>'
        "<strong>MOS打分是评价语音合成系统的重要指标，请根据下面的标准对该系统进行评价：</strong>\n\n"
        '</br></br>'
        "1-无法接受：合成音频质量非常差，难以理解，存在明显的音频失真或不连贯性。\n\n"
        "2-较差：合成音频质量较差，不够自然，存在明显的语音失真或不连贯性。\n\n"
        "3-一般：合成音频质量一般，能够理解，但可能存在一些不自然的音频特征或语音失真。\n\n"
        "4-好：合成音频质量较好，自然度较高，基本上没有明显的语音失真或不连贯性。\n\n"
        "5-非常好：合成音频质量非常好，几乎无法与真实人类语音区分，自然度极高。\n\n"
        '</br></br>'
        f"\n\n请点击[语音合成系统MOS打分](http://192.168.3.145:5000)进行评分"
        '</div>'
    )

    return info_tab


def tts_func(synthesize, language, examples, n_speakers):
    n_speakers = [
        f"spk-{i}" for i in range(n_speakers)] if n_speakers > 0 else ["spk-0"]

    mms_synthesize = gr.Interface(
        fn=synthesize,
        inputs=[
            gr.Text(label="Input text"),
            gr.Dropdown(
                n_speakers,
                label="Speaker",
                value="spk-0",
            ),
            gr.Slider(minimum=0.1, maximum=4.0,
                      value=1.0, step=0.1, label="Speed"),
        ],
        outputs=[
            gr.Audio(label="Generated Audio", type="numpy"),
            gr.Text(label="Filtered text after removing OOVs"),
        ],
        examples=examples,
        title=language,
        description=(
            "Generate audio in your desired language from input text."),
        allow_flagging="never",
    )
    return mms_synthesize


tts = Synthesize()
synth_list = []
lang_list = [
    'Portuguese',
    'Telugu',
    'Hindi',
    'Tamil',
]
# lang_list = [
#     'Genshin',
#     'Chinese',
#     'English',
#     'Arabic',
#     'French',
#     'German',
#     'Italian',
#     'Russian',
#     'Spanish',
#     'Thai',
#     'Japanese',
#     'Korean',
#     'Vietnam',
#     'Dutch',
#     'Portuguese',
#     'Telugu',
#     'Hindi'
# ]

for lang in lang_list:
    synthesize, examples, n_speakers = tts[lang]
    synth_list.append(tts_func(synthesize, lang, examples, n_speakers))

synth_list.append(info())
lang_list.append('打分评测')

tabbed_interface = gr.TabbedInterface(
    synth_list,
    lang_list,
)

demo = gr.Blocks()
with gr.Blocks() as demo:
    gr.HTML("""
        <div
            style="width: 100%; padding-top: 40px; padding-bottom: 20px; background-color: transparent; background-size: cover;"
            <div>
                <div style="margin: 0px 20px;display: flex;">
                    <div class="bili-avatar" style="padding-top: 20px;">
                        <a href="https://cn.timekettle.co" target="_blank">
                        <img style="width:60px;height:60px;border-radius:30px;max-width:60px;" title="前往时空壶"
                            src="https://26349372.s21i.faiusr.com/4/ABUIABAEGAAgmIf5gwYoluervAUwjBE4sRM.png">
                        </a>
                    </div>
                    <div style="margin:20px;color:white">
                        <div style="align-items: flex-end;display: flex">
                            <span style="font-size: 40px;min-width:85px;">VITS语音合成demo</span>
                        </div>
                        <div>
                            <h4 class="h-sign" style="font-size: 15px;">
                                基于<a href="https://github.com/jaywalnut310/vits" target="_blank">VITS</a>技术训练的语音合成demo，支持中英西日韩泰德法意俄阿
                            </h4>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """)
    tabbed_interface.render()
    gr.HTML("""
        <div style="text-align:center">           
            <br/>
            <br/>
            Don't Panic
            <br/>
            仅供学习交流，不可用于或非法用途
            <br/>
            使用本项目模型直接或间接生成的音频，必须声明由AI技术或VITS技术合成
        </div>
    """)


# demo.queue(concurrency_count=6)
demo.launch()
