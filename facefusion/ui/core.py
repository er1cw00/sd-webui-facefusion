
from typing import Dict, Optional, Any, List
from facefusion.ui.typing import Component, ComponentName

UI_COMPONENTS: Dict[ComponentName, Component] = {}

def get_ui_component(name : ComponentName) -> Optional[Component]:
	if name in UI_COMPONENTS:
		return UI_COMPONENTS[name]
	return None


def register_ui_component(name : ComponentName, component: Component) -> None:
	UI_COMPONENTS[name] = component
