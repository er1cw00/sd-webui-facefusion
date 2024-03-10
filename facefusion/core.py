import os

os.environ['OMP_NUM_THREADS'] = '1'

import signal
import ssl
import sys
import warnings
import platform
import shutil
import tempfile
import onnxruntime
from datetime import datetime
from argparse import ArgumentParser, HelpFormatter
from modules import shared, paths

import facefusion.choices
import facefusion.globals
from facefusion import face_analyser, face_masker, content_analyser, metadata, logger, wording
from facefusion.face_analyser import get_one_face, get_average_face
from facefusion.face_store import get_reference_faces, append_reference_face
from facefusion.vision import get_video_frame, detect_fps, read_image, read_static_images
from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.filesystem import is_image, is_video, list_module_names, get_temp_frame_paths, create_temp, move_temp, clear_temp
from facefusion.execution_helper import encode_execution_providers, decode_execution_providers
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.normalizer import normalize_output_path, normalize_padding
from facefusion.ffmpeg import extract_frames, compress_image, merge_video, restore_audio


onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def apply_args() -> None:
    # general
    facefusion.globals.source_paths = None                  #args.source_paths
    facefusion.globals.target_path = None                   #args.target_path
    facefusion.globals.output_path = None                   #normalize_output_path(facefusion.globals.source_paths, facefusion.globals.target_path, args.output_path)
    # misc
    skip_download = getattr(shared.cmd_opts, "facefusion_skip_download", False)
    facefusion.globals.skip_download = skip_download        #args.skip_download
    facefusion.globals.headless = False                     #args.headless
    facefusion.globals.log_level = 'debug'                  #args.log_level
    # execution
    execution_providers = encode_execution_providers(onnxruntime.get_available_providers())
    thread_count = shared.opts.data.get('face_fusion_execution_thread_count', 1)
    queue_count = shared.opts.data.get('face_fusion_execution_queue_count', 1)
    #max_memory = shared.opts.data.get('face_fusion_max_memory', 0)
    facefusion.globals.execution_providers = decode_execution_providers(execution_providers)
    facefusion.globals.execution_thread_count = thread_count       #args.execution_thread_count
    facefusion.globals.execution_queue_count = queue_count         #args.execution_queue_count
    facefusion.globals.max_memory = 0 #max_memory                  #args.max_memory
    # face analyser
    facefusion.globals.face_analyser_order = 'left-right'   #args.face_analyser_order
    facefusion.globals.face_analyser_age = None             #args.face_analyser_age
    facefusion.globals.face_analyser_gender = None          #args.face_analyser_gender
    facefusion.globals.face_detector_model = 'retinaface'   #args.face_detector_model
    facefusion.globals.face_detector_size = '640x640'       #args.face_detector_size
    facefusion.globals.face_detector_score = 0.5            #args.face_detector_score
    # face selector
    facefusion.globals.face_selector_mode = 'reference'     #args.face_selector_mode          
    facefusion.globals.reference_face_position = 0          #args.reference_face_position
    facefusion.globals.reference_face_distance = 0.6        #args.reference_face_distance
    facefusion.globals.reference_frame_number = 0           #args.reference_frame_number
    # face mask
    facefusion.globals.face_mask_types = ['box']                                #args.face_mask_types
    facefusion.globals.face_mask_blur = 0.3                                     #args.face_mask_blur
    facefusion.globals.face_mask_padding = normalize_padding([0, 0, 0, 0])      #normalize_padding(args.face_mask_padding)
    facefusion.globals.face_mask_regions = facefusion.choices.face_mask_regions #args.face_mask_regions
    # frame extraction
    facefusion.globals.trim_frame_start = None              #args.trim_frame_start
    facefusion.globals.trim_frame_end = None                #args.trim_frame_end
    facefusion.globals.temp_frame_format = 'jpg'            #args.temp_frame_format
    facefusion.globals.temp_frame_quality = 100             #args.temp_frame_quality
    facefusion.globals.keep_temp = False                    #args.keep_temp
    # output creation
    output_image_quality = shared.opts.data.get('face_fusion_image_quality', 90)
    output_video_quality = shared.opts.data.get('face_fusion_video_quality', 90)
    output_video_encoder = shared.opts.data.get('face_fusion_video_encoder', "libx264")
    facefusion.globals.output_image_quality = output_image_quality              #args.output_image_quality
    facefusion.globals.output_video_encoder = output_video_encoder              #args.output_video_encoder
    facefusion.globals.output_video_quality = output_video_quality              #args.output_video_quality
    facefusion.globals.keep_fps = False                      #args.keep_fps
    facefusion.globals.skip_audio = False                   #args.skip_audio
    
    # frame processors
    available_frame_processors = list_module_names('processors/frame/modules')
    facefusion.globals.frame_processors = ['face_swapper']     #args.frame_processors
    frame_processors_globals.frame_enhancer_model = 'real_esrgan_x2plus'        #args.frame_enhancer_model
    frame_processors_globals.frame_enhancer_blend = 80                          #args.frame_enhancer_blend
    frame_processors_globals.face_debugger_items = ['kps', 'face-mask']         #args.face_debugger_items
    frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'                 #args.face_enhancer_model
    frame_processors_globals.face_enhancer_blend = 80                           #args.face_enhancer_blend
    frame_processors_globals.face_swapper_model = 'inswapper_128'               #args.face_swapper_model
    facefusion.globals.face_recognizer_model = 'arcface_inswapper'


    proxy_host = getattr(shared.cmd_opts, 'facefusion_proxy', None)
    prongraphic_filtering = shared.opts.data.get('face_fusion_prongraphic_content_filtering', True)
    facefusion.globals.proxy_host = proxy_host
    facefusion.globals.prongraphic_filtering = prongraphic_filtering
    
    flag = shared.opts.data.get('face_fusion_output_path_with_datetime', False)
    default_dir = os.path.join(paths.data_path, "outputs/facefusion")
    output_dir = shared.opts.data.get("face_fusion_output_path", default_dir)
    if flag :
        output_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    facefusion.globals.output_dir = output_dir
    
    tmp_dir = tempfile.gettempdir()
    upload_dir = tmp_dir #shared.opts.data.get("face_fusion_upload_path", tmp_dir)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    facefusion.globals.upload_dir = upload_dir
    
    facefusion.globals.ui_layouts = 'default'                                   #args.ui_layouts
    

def limit_resources() -> None:
    if facefusion.globals.max_memory:
        memory = facefusion.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = facefusion.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes

            kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def pre_check() -> bool:
    model_path = facefusion.globals.model_path
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok=True)

    if sys.version_info < (3, 9):
        logger.error(wording.get('python_not_supported').format(version = '3.9'), __name__.upper())
        return False
    if not shutil.which('ffmpeg'):
        logger.error(wording.get('ffmpeg_not_installed'), __name__.upper())
        return False
    return True

def conditional_process() -> None:
    conditional_append_reference_faces()
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_process('output'):
            return
    if is_image(facefusion.globals.target_path):
        process_image()
    if is_video(facefusion.globals.target_path):
        process_video()

def conditional_append_reference_faces() -> None:
    if 'reference' in facefusion.globals.face_selector_mode and not get_reference_faces():
        source_frames = read_static_images(facefusion.globals.source_paths)
        source_face = get_average_face(source_frames)
        if is_video(facefusion.globals.target_path):
            reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
        else:
            reference_frame = read_image(facefusion.globals.target_path)
        reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
        append_reference_face('origin', reference_face)
        if source_face and reference_face:
            for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
                print(f'frame_processor_module: {frame_processor_module.NAME}')
                reference_frame = frame_processor_module.get_reference_frame(source_face, reference_face, reference_frame)
                reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
                append_reference_face(frame_processor_module.__name__, reference_face)
    

def process_image() -> None:
    if facefusion.globals.prongraphic_filtering and analyse_image(facefusion.globals.target_path):
        return
    shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
    print(f'source: {facefusion.globals.source_paths}')
    print(f'target: {facefusion.globals.target_path}')
    print(f'output: {facefusion.globals.output_path}')
    # process frame
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        logger.info(wording.get('processing'), frame_processor_module.NAME)
        frame_processor_module.process_image(facefusion.globals.source_paths, facefusion.globals.output_path, facefusion.globals.output_path)
        frame_processor_module.post_process()
    # compress image
    logger.info(wording.get('compressing_image'), __name__.upper())
    if not compress_image(facefusion.globals.output_path):
        logger.error(wording.get('compressing_image_failed'), __name__.upper())
    # validate image
    if is_image(facefusion.globals.output_path):
        logger.info(wording.get('processing_image_succeed'), __name__.upper())
    else:
        logger.error(wording.get('processing_image_failed'), __name__.upper())


def process_video() -> None:
    if facefusion.globals.prongraphic_filtering and analyse_video(facefusion.globals.target_path, facefusion.globals.trim_frame_start, facefusion.globals.trim_frame_end):
        return
    fps = detect_fps(facefusion.globals.target_path) if facefusion.globals.keep_fps else 25.0
    # create temp
    logger.info(wording.get('creating_temp'), __name__.upper())
    create_temp(facefusion.globals.target_path)
    # extract frames
    logger.info(wording.get('extracting_frames_fps').format(fps = fps), __name__.upper())
    extract_frames(facefusion.globals.target_path, fps)
    # process frame
    temp_frame_paths = get_temp_frame_paths(facefusion.globals.target_path)
    if temp_frame_paths:
        for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
            logger.info(wording.get('processing'), frame_processor_module.NAME)
            frame_processor_module.process_video(facefusion.globals.source_paths, temp_frame_paths)
            frame_processor_module.post_process()
    else:
        logger.error(wording.get('temp_frames_not_found'), __name__.upper())
        return
    # merge video
    logger.info(wording.get('merging_video_fps').format(fps = fps), __name__.upper())
    if not merge_video(facefusion.globals.target_path, fps):
        logger.error(wording.get('merging_video_failed'), __name__.upper())
        return
    # handle audio
    if facefusion.globals.skip_audio:
        logger.info(wording.get('skipping_audio'), __name__.upper())
        move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
    else:
        logger.info(wording.get('restoring_audio'), __name__.upper())
        if not restore_audio(facefusion.globals.target_path, facefusion.globals.output_path):
            logger.warn(wording.get('restoring_audio_skipped'), __name__.upper())
            move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
    # clear temp
    logger.info(wording.get('clearing_temp'), __name__.upper())
    clear_temp(facefusion.globals.target_path)
    # validate video
    if is_video(facefusion.globals.output_path):
        logger.info(wording.get('processing_video_succeed'), __name__.upper())
    else:
        logger.error(wording.get('processing_video_failed'), __name__.upper())

    
def facefusion_init() -> None:
    apply_args()
    logger.init(facefusion.globals.log_level)
    #limit_resources()
    if (not pre_check()
        or not content_analyser.pre_check()
        or not face_analyser.pre_check() 
        or not face_masker.pre_check()):
        return
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_check():
            return
    return
