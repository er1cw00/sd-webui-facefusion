import os
import json
import gradio as gr
import facefusion
from pathlib import Path
from modules import script_callbacks, shared,scripts
from facefusion import wording
from facefusion.typing import OutputVideoEncoder
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
    common_options.listen()
    source.listen()
    target.listen()
    output.listen()
    preview.listen()
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
    shared.opts.add_option("face_fusion_execution_thread_count", 
                           shared.OptionInfo(
                               default=4, 
                               label=wording.get('execution_thread_count_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 1, "maximum": 128, "step": 1}, 
                               refresh=update_execution_thread_count,
                               section=section
                           ))
    shared.opts.add_option("face_fusion_execution_queue_count", 
                           shared.OptionInfo(
                               default=1, 
                               label=wording.get('execution_queue_count_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 1, "maximum": 32, "step": 1}, 
                               refresh=update_execution_queue_count,
                               section=section
                           ))
    shared.opts.add_option("face_fusion_max_memory", 
                           shared.OptionInfo(
                               default=1, 
                               label=wording.get('max_memory_help'), 
                               component=gr.Slider, 
                               component_args={"minimum": 0, "maximum": 128, "step": 1}, 
                               refresh=update_execution_max_memory,
                               section=section
                           ))
    shared.opts.add_option("face_fusion_prongraphic_content_filtering",
                           shared.OptionInfo(
                               default=True,
                               label="Enable adult content filtering",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               refresh=update_prongraphic_content_filtering,
                               section=section))
    shared.opts.add_option("face_fusion_save_folder",
                        shared.OptionInfo(
                            default="face-fusion",
                            label="Folder name where output images or videos will be saved",
                            component=gr.Radio,
                            component_args={"choices": ["face-fusion", "img2img-images"]},
                            section=section))
    shared.opts.add_option("face_fusion_image_quality", 
                           shared.OptionInfo(
                               default=90, 
                               label=wording.get('output_image_quality_slider_label'), 
                               component=gr.Slider, 
                               component_args={"minimum": 0, "maximum": 100, "step": 1}, 
                               refresh=update_output_image_quality,
                               section=section))
    shared.opts.add_option("face_fusion_video_quality", 
                           shared.OptionInfo(
                               default=90, 
                               label=wording.get('output_video_quality_slider_label'), 
                               component=gr.Slider, 
                               component_args={"minimum": 0, "maximum": 100, "step": 1}, 
                               refresh=update_output_video_quality,
                               section=section))
    shared.opts.add_option("face_fusion_video_encoder", 
                           shared.OptionInfo(
                               default='libx264', 
                               label=wording.get('output_video_encoder_dropdown_label'), 
                               component=gr.Dropdown, 
                               component_args=lambda: {"choices": facefusion.choices.output_video_encoders},
                               refresh=update_output_video_encoder,
                               section=section))
       
    
def update_execution_thread_count() -> None:
    thread_count = shared.opts.data.get('face_fusion_execution_thread_count', 1)
    facefusion.globals.execution_thread_count = thread_count
    
def update_execution_queue_count() -> None:
    quene_count = shared.opts.data.get('face_fusion_execution_queue_count', 1)
    facefusion.globals.execution_queue_count = quene_count
    
def update_execution_max_memory() -> None:
    max_memory = shared.opts.data.get('face_fusion_max_memory', 0)
    facefusion.globals.max_memory = max_memory  
    
def update_prongraphic_content_filtering() -> None:
    prongraphic_filtering = shared.opts.data.get('face_fusion_prongraphic_content_filtering', True)
    facefusion.globals.prongraphic_filtering = prongraphic_filtering
    
def update_output_image_quality() -> None:
    output_image_quality = shared.opts.data.get('face_fusion_image_quality', 90)
    facefusion.globals.output_image_quality = output_image_quality

def update_output_video_quality() -> None:
    output_video_quality = shared.opts.data.get('face_fusion_video_quality', 90)
    facefusion.globals.output_video_quality = output_video_quality
    
def update_output_video_encoder() -> None:
    output_video_encoder = shared.opts.data.get('face_fusion_video_encoder', 90)
    facefusion.globals.output_video_encoder = output_video_encoder


