import os
import json
import gradio as gr
from pathlib import Path
from modules import script_callbacks, shared,scripts
from facefusion import wording
from facefusion.ui.components import about,frame_processors,frame_processors_options,execution,limit_resources,common_options,output_options
from facefusion.ui.components import source,target,output
from facefusion.ui.components import preview,trim_frame,face_selector,face_masker,face_analyser


# Webui root path
FACE_FUSION = 'FaceFusion'
ROOT_DIR = Path().absolute()
__version__ = "1.0.0"


def render() :
    with gr.Row():
        with gr.Column(scale = 2):
            with gr.Blocks():
                about.render()
            with gr.Blocks():
                frame_processors.render()
                frame_processors_options.render()
            with gr.Blocks():
                execution.render()
            with gr.Blocks():
                limit_resources.render()
            with gr.Blocks():
                output_options.render()
            with gr.Blocks():
                common_options.render()
        with gr.Column(scale = 2):
            with gr.Blocks():
                source.render()
            with gr.Blocks():
                target.render()
            with gr.Blocks():
                output.render()
        with gr.Column(scale = 3):
            with gr.Blocks():
                preview.render()
            with gr.Blocks():
                trim_frame.render()
            with gr.Blocks():
                face_selector.render()
            with gr.Blocks():
                face_masker.render()
            with gr.Blocks():
                face_analyser.render()

def listen() :
    frame_processors.listen()
    frame_processors_options.listen()
    execution.listen()
    limit_resources.listen()
    output_options.listen()
    common_options.listen()
    source.listen()
    target.listen()
    output.listen()
    trim_frame.listen()
    face_selector.listen()
    face_masker.listen()
    face_analyser.listen()
    
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as layout:
        render()
        listen()
    return [(layout, "FaceFusion", "facefusion")]

def on_ui_settings():
    section = ("face fusion", "Face Fusion")
    shared.opts.add_option("face_fusion_thread_count", 
                           shared.OptionInfo(
                               default=4, 
                               label=wording.get('execution_thread_count_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 1, "maximum": 128, "step": 1}, 
                               section=section
                           ))
    shared.opts.add_option("face_fusion_queue_count", 
                           shared.OptionInfo(
                               default=1, 
                               label=wording.get('execution_queue_count_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 1, "maximum": 32, "step": 1}, 
                               section=section
                           ))
    shared.opts.add_option("face_fusion_max_memory", 
                           shared.OptionInfo(
                               default=1, 
                               label=wording.get('max_memory_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 0, "maximum": 128, "step": 1}, 
                               section=section
                           ))
    shared.opts.add_option("face_fusion_save_folder",
                        shared.OptionInfo(
                            default="face-fusion",
                            label="Folder name where output images or videos will be saved",
                            component=gr.Radio,
                            component_args={"choices": ["face-fusion", "img2img-images"]},
                            section=section))
    shared.opts.add_option("face_fusion_prongraphic_content_filtering",
                           shared.OptionInfo(
                               default=True,
                               label="Enable adult content filtering",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))