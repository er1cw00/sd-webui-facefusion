import os 
import gradio as gr
import onnxruntime
from typing import Any, Optional, List
from fastapi import FastAPI, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from modules import shared
from modules.api import api
from modules.api.models import *
from modules.call_queue import queue_lock
from facefusion import metadata, logger, wording, choices, globals
from facefusion.core import conditional_process
from facefusion.normalizer import normalize_output_path, check_output_path
from facefusion.filesystem import list_module_names, is_file
from facefusion.execution_helper import encode_execution_providers, decode_execution_providers
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.processors.frame import globals as frame_processors_globals, choices as frame_processors_choices

class FrameProcessorRequest(BaseModel):
    name: str
    model: str
    blend: float
    debug_items: Optional[List[str]]
        
class FrameProcessRequest(BaseModel):
    sources: List[str]
    target: str
    output: Optional[str]
    providers: List[str]
    processors: List[FrameProcessorRequest]
    face_selector: str = 'reference'
    face_refer_distance: float = 0.6
    face_mask_types: List[str] = ['box']
    face_mask_blur: float = 0.3
    face_analyse_order:str = 'left-right'
    face_analyse_age:Optional[str] = None
    face_analyse_gender:Optional[str] = None
    face_detect_model: str = 'retinaface'
    face_detect_score: float = 0.5
    face_detect_size: str = '640x640'
    skip_download: Optional[bool] = None
        
        
class FrameProcessResponse(BaseModel):
    output: Optional[str]
    detail: Optional[str]
    
def facefusion_api(_: gr.Blocks, app: FastAPI):
    
    @app.get("/facefusion/version")
    async def version():
        version = metadata.get('version')
        return {"version": version}

    @app.get("/facefusion/exection_providers")
    async def exection_providers():
        providers = encode_execution_providers(onnxruntime.get_available_providers())
        return {'providers': providers}
    
    @app.get("/facefusion/frame_processors")
    async def frame_processors():
        frame_processors = list_module_names('processors/frame/modules')
        logger.debug(f'frame_processors: {frame_processors}', __name__.upper())
        result = []
        print(f'result: {result}')
        for frame_processor in frame_processors:
            processer = {'processor': frame_processor}
            if frame_processor == 'face_swapper':
                processer['models'] = frame_processors_choices.face_swapper_models
            elif frame_processor == 'face_enhancer':
                processer['models'] = frame_processors_choices.face_enhancer_models
            elif frame_processor == 'face_debugger':
                processer['models'] = frame_processors_choices.face_debugger_items
            elif frame_processor == 'frame_enhancer':
                processer['models'] = frame_processors_choices.frame_enhancer_models
            print(f'processer: {processer}')
            result.append(processer)
        return result

    @app.get("/facefusion/download")
    async def download(filename:str):
        if len(filename) == 0:
            raise HTTPException(status_code=400, detail=f'Bad Request')
        path = os.path.join(globals.output_dir, filename)
        if not is_file(path):
            raise HTTPException(status_code=404, detail=f'File Not Found')
        return FileResponse(path, filename=filename)
    
    @app.post("/facefusion/frame_process")
    async def process(req: FrameProcessRequest = Body(...)) -> FrameProcessResponse:
        print(f'frame_process >> \n{req}')
        success, message = set_parameter(req)
        if not success:
            raise HTTPException(status_code=422, detail=message)
        globals.output_path = normalize_output_path(globals.target_path, globals.output_path)
        print(f'processors: {globals.frame_processors}')
        conditional_process()      
        response = FrameProcessResponse(output=globals.output_path)
        globals.output_path = None
        print(f'frame_process <<')
        return response
        
    
    def set_parameter(req: FrameProcessRequest):
        sources = []
        for source in req.sources:
            if source.startswith('file://'):
                path = source[7:]
                if is_file(path):
                    sources.append(path)
        if len(sources) == 0:
            return (False, 'No source images')
        globals.source_paths = sources
        if req.target.startswith('file://'):
            target_path = req.target[7:]
            print(f'target:{target_path}')
            if not is_file(target_path):
                return (False, 'No target image or video')
            globals.target_path = target_path
        else:
            return (False, 'No target image or video')
        if req.output.startswith('file://'):
            req.output = req.output[7:]
        if not check_output_path(globals.target_path, req.output):
            return (False, "Unknown output path")
        globals.output_path = req.output
        req.providers = decode_execution_providers(req.providers)
        if len(req.providers) == 0:
            return (False, 'No executor providers')
        globals.exection_providers = req.providers
        if req.face_selector not in choices.face_selector_modes:
            return (False, 'Unknown face selector')
        globals.face_selector_mode = req.face_selector
        if req.face_refer_distance < 0.0 and req.face_refer_distance > 1.5:
            return (False, 'Unknown face reference distable')
        globals.reference_face_distance = req.face_refer_distance
        req.face_mask_types = [t for t in req.face_mask_types if t in choices.face_mask_types] 
        if len(req.face_mask_types) == 0:
            return (False, 'Unknown face mask type')
        globals.face_mask_types = req.face_mask_types
        if req.face_mask_blur < 0.0 and req.face_mask_blur >= 1.0:
            return (False, 'Unknown face mask blur')
        globals.face_mask_blur = req.face_mask_blur 
        if req.face_analyse_order not in choices.face_analyser_orders:
            return (False, 'Unknown face analyse order')
        globals.face_analyser_order = req.face_analyse_order
        if req.face_analyse_age != None and req.face_analyse_age not in choices.face_analyser_ages:
            return (False, 'Unknown face analyse age')
        globals.face_analyser_age = req.face_analyse_age
        if req.face_analyse_gender != None and req.face_analyse_gender not in choices.face_analyser_genders:
            return (False, 'Unknown face analyse gender')
        globals.face_analyser_gender = req.face_analyse_gender
        if req.face_detect_model not in choices.face_detector_models:
            return (False, 'Unknown face detect model')
        globals.face_detector_model = req.face_detect_model
        if req.face_detect_score < 0 and req.face_detect_score > 1:
            return (False, 'Unknown face detect score')
        globals.face_detector_score = req.face_detect_score
        if req.face_detect_size not in choices.face_detector_sizes:
            return (False, 'Unknown face detect size')
        globals.face_detector_size = req.face_detect_size

        globals.skip_download = req.skip_download
        if len(req.processors) == 0:
            return (False, 'No Frame Prosessor')
        processors = []
        for processor in req.processors:
            if processor.name == 'face_swapper':
                processors.append(processor.name)
                if processor.model not in frame_processors_choices.face_swapper_models:
                    return (False, 'Unknown face_swaper model: {process.model}')
                frame_processors_globals.face_swapper_model = processor.model 
                print(f'face swapper model: {frame_processors_globals.face_swapper_model}')
            elif processor.name == 'face_enhancer':
                processors.append(processor.name)
                if processor.model not in frame_processors_choices.face_enhancer_models:
                    return (False, 'Unknown face_enhancer model: {process.model}')
                if processor.blend < 0 and processor.blend > 100:
                    return (False, 'Unknown frame enhance blend')
                frame_processors_globals.face_enhancer_model = processor.model       #args.frame_enhancer_model
                frame_processors_globals.face_enhancer_blend = processor.blend
            elif processor.name == 'face_debugger':
                processors.append(processor.name)
                if processor.model not in frame_processors_choices.frame_enhancer_models:
                    return (False, 'Unknown frame_enhancer model: {process.model}')
                debug_items = [item for item in processor.debug_items if item in frame_processors_choices.face_debugger_items] 
                if len(debug_items) == 0:
                    return (False, 'No Debug Item')
                frame_processors_globals.face_debugger_items = debug_items
            elif processor.name == 'frame_enhancer':
                processors.append(processor.name)
                if processor.model not in frame_processors_choices.frame_enhancer_models:
                    return (False, 'Unknown frame_enhancer model: {process.model}')
                if processor.blend < 0 and processor.blend > 100:
                    return (False, 'Unknown frame enhance blend')
                frame_processors_globals.frame_enhancer_model = processor.model
                frame_processors_globals.frame_enhancer_blend = processor.blend
            else:    
                return (False, 'Unknown frame processor: {process.name}')
            globals.frame_processors = processors;
        return (True, '')

            
