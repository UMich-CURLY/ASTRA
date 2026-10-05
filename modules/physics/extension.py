# MIT License
# 
# Copyright (c) 2024 <COPYRIGHT_HOLDERS>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

import asyncio
import gc

import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
import omni.ui as ui
import omni.usd
import omni.kit.commands as commands
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription, make_menu_item_description
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from omni.usd import StageEventType
from pxr import UsdShade

# Local imports
from .global_variables import EXTENSION_TITLE
from .ui_builder import UIBuilder
from .ocean_waves import Water
from .asv import ASVController
from .gps_sensor import GPSSensor
from .simple_floating_objects import FoatingObjects

"""
Manages the extension. The first four functions on_startup(), on_shutdown(), _on_window(), and _build_ui()
are boilerplate code from NVIDIA. Only modify if absolutely necessary (is it?)
"""

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """
        Called by the Extension Manager when the extension is being started.
        https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/classomni_1_1ext_1_1IExt.html
        """
        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

        # Build Window
        self._window = ScrollingWindow(title=EXTENSION_TITLE, width=600, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM)
        self._window.set_visibility_changed_fn(self._on_window)

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            ext_id,
            f"CreateUIExtension:{EXTENSION_TITLE}",
            self._menu_callback,
            description=f"Add {EXTENSION_TITLE} Extension to UI toolbar",
        )
        self._menu_items = [
            MenuItemDescription(name='Settings', onclick_action=(ext_id, f"CreateUIExtension:{EXTENSION_TITLE}"))
        ]

        add_menu_items(self._menu_items, EXTENSION_TITLE)

        # Initialize Classes
        self._intialize_classes()

        # Intialize Extension
        self._initialize_ext()

        # Build UI
        self._build_ui()

        # Events
        self._usd_context = omni.usd.get_context()
        self._physxIFace = _physx.get_physx_interface()
        self._physx_subscription = None
        self._stage_event_sub = None
        self._timeline = omni.timeline.get_timeline_interface()

        # Subscribe to Stage and Timeline Events
        self._usd_context = omni.usd.get_context()
        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)
        stream = self._timeline.get_timeline_event_stream()
        self._timeline_event_sub = stream.create_subscription_to_pop(self._on_timeline_event)

        print(f"{EXTENSION_TITLE}: fully intialized")

    def on_shutdown(self):
        """
        Called by the Extension Manager when the extension is shutdown.
        https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/classomni_1_1ext_1_1IExt.html
        """
        self._models = {}
        remove_menu_items(self._menu_items, EXTENSION_TITLE)

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self.ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")

        if self._window:
            self._window = None
        self._cleanup()
        gc.collect()

    def _on_window(self, visible):
        """Called one time when the extension's window visibility is changed"""
        if self._window.visible:
            # # Subscribe to Stage and Timeline Events
            # self._usd_context = omni.usd.get_context()
            # events = self._usd_context.get_stage_event_stream()
            # self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)
            # stream = self._timeline.get_timeline_event_stream()
            # self._timeline_event_sub = stream.create_subscription_to_pop(self._on_timeline_event)

            self._build_ui()
        else:
            # self._usd_context = None
            # self._stage_event_sub = None
            # self._timeline_event_sub = None
            self._cleanup()

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self._build_extension_ui()

        async def dock_window():
            await omni.kit.app.get_app().next_update_async()

            def dock(space, name, location, pos=0.5):
                window = omni.ui.Workspace.get_window(name)
                if window and space:
                    window.dock_in(space, location, pos)
                return window

            tgt = ui.Workspace.get_window("Viewport")
            dock(tgt, EXTENSION_TITLE, omni.ui.DockPosition.LEFT, 0.33)
            await omni.kit.app.get_app().next_update_async()

        self._task = asyncio.ensure_future(dock_window())


    #################################################################
    # Functions below this point call user functions
    #################################################################


    def _initialize_ext(self):
        
        pass


    def _intialize_classes(self):
        self.ui_builder = UIBuilder()
        self.water = Water()
        self.asv_controller = ASVController()
        self.gps_sensor = GPSSensor()
        self.foating_objects = FoatingObjects()
    

    def _cleanup(self):
        self.ui_builder.cleanup()
        self.asv_controller.shutdown()
        self.gps_sensor.shutdown()


    def _menu_callback(self):
        """
        Called when the menu item at the top is clicked
        """
        self._window.visible = not self._window.visible
        self.ui_builder.on_menu_callback()


    def _on_timeline_event(self, event):
        # https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/omni.timeline/omni.timeline.TimelineEventType.html
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if not self._physx_subscription:
                self._physx_subscription = self._physxIFace.subscribe_physics_step_events(self._on_physics_step)
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._physx_subscription = None

        self.ui_builder.on_timeline_event(event)
        self.water.on_timeline_event(event)
        self.asv_controller.on_timeline_event(event)
        self.gps_sensor.on_timeline_event(event)
        self.foating_objects.on_timeline_event(event)


    def _on_physics_step(self, step):
        self.ui_builder.on_physics_step(step)
        self.water.on_physics_step(step)
        self.asv_controller.on_physics_step(step)
        self.gps_sensor.on_physics_step(step)
        self.foating_objects.on_physics_step(step)


    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED) or event.type == int(StageEventType.CLOSED):
            # stage was opened or closed, cleanup
            self._physx_subscription = None
            self.ui_builder.cleanup()

        self.ui_builder.on_stage_event(event)


    def _build_extension_ui(self):
        # Call user function for building UI
        self.ui_builder.build_ui()
