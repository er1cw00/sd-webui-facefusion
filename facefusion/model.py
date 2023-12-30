from typing import List, Dict

from modules.api import models as sd_models
from pydantic import BaseModel, Field


class FaceFusionImageRequest(BaseModel):
    source: str = Field(default="", title="Image", description="Face Image to work on, must be a Base64 string containing the image's data.")
    target: str = Field(default="", title="Image", description="Target Image to be replace, must be a Base64 string containing the image's data.")
   
#class  FaceFusionImageResponse(BaseModel):