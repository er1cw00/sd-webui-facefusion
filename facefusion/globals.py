import os
from typing import List, Optional
from facefusion.filesystem import resolve_relative_path
from facefusion.typing import LogLevel, FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, FaceMaskType, FaceMaskRegion, OutputVideoEncoder, FaceDetectorModel, FaceRecognizerModel, TempFrameFormat, Padding

# general
source_paths : Optional[List[str]] = None
target_path : Optional[str] = None
output_path : Optional[str] = None
output_dir : Optional[str] = None 
upload_dir : Optional[str] = None 
output_image_format : Optional[str] = 'jpg'
# misc
watermark : Optional[bool] = None
watermark_logo_path: Optional[str] = None
skip_download : Optional[bool] = None
headless : Optional[bool] = None
log_level : Optional[LogLevel] = None
proxy_host : Optional[str] = None
# execution
execution_providers : List[str] = []
execution_thread_count : Optional[int] = None
execution_queue_count : Optional[int] = None
max_memory : Optional[int] = None
# face analyser
face_analyser_order : Optional[FaceAnalyserOrder] = None
face_analyser_age : Optional[FaceAnalyserAge] = None
face_analyser_gender : Optional[FaceAnalyserGender] = None
face_detector_model : Optional[FaceDetectorModel] = None
face_detector_size : Optional[str] = None
face_detector_score : Optional[float] = None
face_recognizer_model : Optional[FaceRecognizerModel] = None
# face selector
face_selector_mode : Optional[FaceSelectorMode] = None
reference_face_position : Optional[int] = None
reference_face_distance : Optional[float] = None
reference_frame_number : Optional[int] = None
# face mask
face_mask_types : Optional[List[FaceMaskType]] = None
face_mask_blur : Optional[float] = None
face_mask_padding : Optional[Padding] = None
face_mask_regions : Optional[List[FaceMaskRegion]] = None
# frame extraction
trim_frame_start : Optional[int] = None
trim_frame_end : Optional[int] = None
temp_frame_format : Optional[TempFrameFormat] = None
temp_frame_quality : Optional[int] = None
keep_temp : Optional[bool] = None
# output creation
output_image_quality : Optional[int] = None
output_video_encoder : Optional[OutputVideoEncoder] = None
output_video_quality : Optional[int] = None
keep_fps : Optional[bool] = None
skip_audio : Optional[bool] = None
prongraphic_filtering : Optional[bool] = True
# frame processors
frame_processors : List[str] = []
# uis
ui_layouts : List[str] = []

model_path: str = resolve_relative_path("../models") #os.path.join(os.path.dirname(os.path.realpath(__file__)),