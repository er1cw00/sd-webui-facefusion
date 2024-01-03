from typing import List, Optional
import os
import facefusion
from datetime import datetime
from facefusion.filesystem import is_file, is_directory, is_image, is_video
from facefusion.typing import Padding

def check_output_path(target_path : str, output_path: str):
	if output_path == None:
		return False
	dir, filename = os.path.split(output_path)
	if not os.path.isdir(dir):
		return False
	_, target_extension = os.path.splitext(os.path.basename(target_path))
	_, output_extension = os.path.splitext(filename)
	if target_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
		if output_extension not in ['.jpg', '.jpeg', '.png']:
			return False
	elif target_extension in ['.mp4', '.ogg', '.avi', '.mpg', '.mov']:
		if output_extension not in ['.mp4', '.mov']:
			return False
	return True

def normalize_output_path(target_path : str, output_path: str) -> Optional[str]:
	if not is_file(target_path) or check_output_path(target_path, output_path):
		return output_path
	filename = None
	now = datetime.now()
	if is_image(target_path):
		image_suffix = facefusion.globals.output_image_format
		filename = f'image_{int(now.timestamp() * 1000)}.{image_suffix}'
	elif is_video(target_path):
		filename = f'video_{int(now.timestamp() * 1000)}.mp4'
	if filename != None:
		output_path = os.path.join(facefusion.globals.output_dir, filename)
	return output_path


def normalize_padding(padding : Optional[List[int]]) -> Optional[Padding]:
	if padding and len(padding) == 1:
		return tuple([ padding[0], padding[0], padding[0], padding[0] ]) # type: ignore[return-value]
	if padding and len(padding) == 2:
		return tuple([ padding[0], padding[1], padding[0], padding[1] ]) # type: ignore[return-value]
	if padding and len(padding) == 3:
		return tuple([ padding[0], padding[1], padding[2], padding[1] ]) # type: ignore[return-value]
	if padding and len(padding) == 4:
		return tuple(padding) # type: ignore[return-value]
	return None
