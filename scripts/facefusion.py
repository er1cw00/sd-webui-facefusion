from facefusion.ui.layout import on_ui_tabs,on_ui_settings
from facefusion.api import facefusion_api
from facefusion.core import facefusion_init
from facefusion import metadata

#from facefusion.api import api
from modules import script_callbacks

facefusion_init()

script_callbacks.on_app_started(facefusion_api)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)

name = metadata.get('name')
version = metadata.get('version')
print(f'[-] sd-webui-facefusion initialized. {name} {version}')