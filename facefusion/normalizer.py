from typing import List, Optional
import os
import facefusion
from datetime import datetime
from facefusion.filesystem import is_file, is_directory, is_image, is_video
from facefusion.typing import Padding

def normalize_output_path(target_path : str) -> Optional[str]:
	if not is_file(target_path):
		return None
	output_path = None
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
