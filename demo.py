import os
import torch
from PIL import Image
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (
    MODEL_7B_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)
from typing import List, Tuple
import gradio as gr

def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    device = tokens.device
    tokens = tokens[0]
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments

def generate_response(model,instruction: str,image,task,radio_img,
                      
            text_temp,text_top_p,text_greedy,text_repetition_penalty,
            guidance_scale_text,guidance_scale_image,
            img_temp,img_top_p,img_greedy,):
    # Load Chameleon model
    
    options = Options()
    options.txt.temp = text_temp
    options.txt.top_p = text_top_p
    options.txt.greedy = text_greedy
    options.txt.repetition_penalty = text_repetition_penalty
    options.img.cfg.guidance_scale_image = guidance_scale_image
    options.img.cfg.guidance_scale_text = guidance_scale_text
    options.img.temp = img_temp
    options.img.top_p = img_top_p
    options.img.greedy = img_greedy
    # options.txt.greedy=True
    batch_prompt_ui = [[]]
    assert radio_img in ['prepend','append']
    if image and radio_img == 'prepend':
         batch_prompt_ui[0] += [{"type": "image", "value": f"file:{image}"},]
    batch_prompt_ui[0] += [{"type": "text", "value": instruction}]
    if image and radio_img == 'append':
         batch_prompt_ui[0] += [{"type": "image", "value": f"file:{image}"},]
    batch_prompt_ui[0] += [{"type": "sentinel", "value": "<END-OF-TURN>"}]
    print(f"DEBUG: Task is {task}")
    if task == 'text-gen':
        options.img = False
    
    if task == 'image-gen':
        options.txt = False
        batch_prompt_ui[0] += [{"type": "sentinel", "value": "<START-OF-IMAGE>"}]

    tokens: torch.LongTensor = model.generate(batch_prompt_ui=batch_prompt_ui, options=options)
    
    if task == 'image-gen':
        segments = [('image_seg', tokens.reshape(1, -1))]
    else:
        # split
        boi, eoi = model.vocab.begin_image, model.vocab.end_image   # 8197(boi), 8196(eoi)
        segments = split_token_sequence(tokens, boi, eoi)


    responses = []
    save_dir = "./outputs/interleaved/"
    os.makedirs(save_dir, exist_ok=True)
    
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            img: Image = model.decode_image(seg_tokens)[0]
            # image_path = os.path.join(save_dir, f"{seg_id}.png")
            # img.save(image_path)
            responses.append({"type": "image", "value": img})
        else:
            decoded_text = model.decode_text(seg_tokens)[0]
            responses.append({"type": "text", "value": decoded_text})
    print(responses)
    return responses
import base64,io
def do_chatbot(message, chat_history,img,radio,radio_img,
        text_temp,text_top_p,text_greedy,text_repetition_penalty,
            guidance_scale_text,guidance_scale_image,
            img_temp,img_top_p,img_greedy,
               model=None,
               ):
    history=chat_history
    input_text = message # message['text']
    
    if img and radio_img == 'prepend':
        history.append(dict(role="user", content=dict(path=img)))
    history.append(dict(role="user", content=input_text))
    if img and radio_img == 'append':
        history.append(dict(role="user", content=dict(path=img)))
    print("Starting Generation...")
    responses = generate_response(model,input_text,img,radio,radio_img,
                                          text_temp,text_top_p,text_greedy,text_repetition_penalty,
            guidance_scale_text,guidance_scale_image,
            img_temp,img_top_p,img_greedy,
                                  
                                  )
    print("Generation Finished!")
    for response in responses:
        if response["type"] == "text":
            history.append(dict(role="assistant", content=str(response['value'])))
        elif response["type"] == "image":
            
            buffer = io.BytesIO()
            response['value'].save(buffer, format='PNG')
            buffer.seek(0)
            # Embed in HTML (base64 data URI)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            html_image_tag = f'<img src="data:image/png;base64,{image_base64}" alt="PIL Image">'
            history.append(dict(role="assistant", content=gr.HTML(html_image_tag)))  # Add image path to chat history
    # return history
    return "",history,None

# Using ChatInterface

import argparse
import time
from functools import partial
from pathlib import Path
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",'-c',type=str,default='ckpts')
    parser.add_argument("--debug",action='store_true')
    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)
    
    MODEL_7B_PATH = ckpt_path / "models" / "7b"
    TOKENIZER_TEXT_PATH = ckpt_path / "tokenizer" / "text_tokenizer.json"
    TOKENIZER_IMAGE_PATH = ckpt_path / "tokenizer" / "vqgan.ckpt"
    TOKENIZER_IMAGE_CFG_PATH = ckpt_path / "tokenizer" / "vqgan.yaml"

    if args.debug:
        # Test GUI Only
        model = None
    else:
        model = ChameleonInferenceModel(
            MODEL_7B_PATH.as_posix(),
            TOKENIZER_TEXT_PATH.as_posix(),
            TOKENIZER_IMAGE_CFG_PATH.as_posix(),
            TOKENIZER_IMAGE_PATH.as_posix(),
        )


    with gr.Blocks() as demo:
        with gr.Row():
            head=gr.HTML('<div><h1>MedMax</h1></div>')
        with gr.Row():
            info=gr.Markdown(''' 
                            code: [Link](https://github.com/Hritikbansal/medmax)
                            
                            project: [Link](https://mint-medmax.github.io/)
                            
                            data: [Link](https://huggingface.co/datasets/mint-medmax/medmax_data)
                             ''')
        with gr.Row():
            disclaimer=gr.Markdown('''
                           This Model (MedMax) is intended for informational and educational purposes only and does not provide medical advice, diagnosis, or treatment. It is designed to assist users by generating content based on input provided and should not be used as a substitute for professional medical advice, expertise, or consultation.
                            
                            Users should always seek the guidance of qualified healthcare professionals with any questions or concerns they may have regarding medical conditions or treatments. This tool is not a replacement for licensed medical practitioners and should not be relied upon to make critical health decisions.
                             ''')
            
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="filepath",label="Input Image")
                with gr.Tab("Basic Config"):
                    radio = gr.Radio(["any-gen", "text-gen", "image-gen"], label="Generation Mode",value='any-gen',
                                     info="Determines whether the output should contains both image and text, or only one of them.")
                    
                with gr.Tab("Advanced Config"):
                    with gr.Tab("Input"):
                        radio_img = gr.Radio(["prepend", "append"], label="Image Input Mode",value='prepend',info="prepend meas image comes before text, append means image comes after the text.")
                    with gr.Tab("Text"):
                        text_temp = gr.Slider(0,1.5,value=1.2,step=0.1,label="Temperature")
                        text_top_p = gr.Slider(0,1.0,value=0.9,step=0.1,label="Top P")
                        text_greedy = gr.Checkbox(value=True,label="Greedy")
                        text_repetition_penalty = gr.Slider(0,1.5,value=1.2,step=0.1,label="Repetition Penalty")
                        
                    with gr.Tab("CFG"):
                        guidance_scale_text = gr.Slider(0,5.0,value=3.0,step=0.1,label="Text Guidance Scale")
                        guidance_scale_image = gr.Slider(0,5.0,value=1.2,step=0.1,label="Image Guidance Scale")
                        
                    with gr.Tab("Image"):
                        img_temp = gr.Slider(0,1.5,value=0.7,step=0.1,label="Temperature")
                        img_top_p = gr.Slider(0,1.0,value=0.9,step=0.1,label="Top P")
                        img_greedy = gr.Checkbox(value=False,label="Greedy")
    
                
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(type="messages")
                msg = gr.Textbox(label="Input",info="press enter to submit")
            
        with gr.Row():  
            clear = gr.ClearButton([msg, chatbot,img])
        msg.submit(partial(do_chatbot,model=model), [msg, chatbot,img,radio,radio_img,
                                                     text_temp,text_top_p,text_greedy,text_repetition_penalty,
                                                     guidance_scale_text,guidance_scale_image,
                                                     img_temp,img_top_p,img_greedy
                                                     ], [msg, chatbot,img])
        examples = gr.Examples(
        examples=[
            [None, "Generate a CT image of the abdomen highlighting a large, irregular, homogenous lesion approximately 9cm in size, originating from the lesser curvature of the stomach. Ensure the lesion is well-defined and distinct from the surrounding gastric tissue, providing a clear view of the stomach and the abnormal growth for detailed assessment.", "image-gen", "prepend"],
            # [None, "generate a image of brain ct-scan", "any-gen", "append"],
            [None, "Could you explain what makes glomus tumors unique in their cellular structure?", "any-gen", "prepend"],
            ['assets/demo_3.png', "Examine this medical image and document your observations in a standard clinical report format.", "text-gen", "prepend"],
            ['assets/demo_2.png', "Provide a brief overview of what is shown in the image", "text-gen", "prepend"],
            ['assets/demo_1.png', "What is the arrow pointing to", "text-gen", "append"],
            ['assets/demo_4.png', "What can be observed in this image?\nA: Meniscal abnormality B: Bone fracture\nC: Ligament tear\nD: Cartilage erosion", "text-gen", "prepend"]
        ],
        inputs=[img,msg,radio,radio_img],
        )
    from huggingface_hub import HfApi
    
    _,_,url = demo.launch(show_api=True,share=True,prevent_thread_lock=True)
    if not args.debug:
        repo_id = "mint-medmax/medmax-demo-v1.0"
        api = HfApi()
        api.add_space_variable(repo_id,'DEMO_URL',url)
    while True:
        time.sleep(1)
