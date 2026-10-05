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

"""
Omniverse UI Framework:
  https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/Overview.html

Isaac Sim UI Utilities extension:
  https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.ui/docs/index.html
"""

from numpy import inf

import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, Frame, StateButton, CheckBox, UIWidgetWrapper, FloatField
from isaacsim.gui.components.ui_utils import get_style, combo_floatfield_slider_builder, xyz_builder
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

# Local imports
from .ocean_waves import Water
from .global_variables import EXTENSION_TITLE


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []

        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Water defaults
        self.default_enable_waves = False
        self.default_wave_scale = 1.0
        self.default_directional_spreading = 0.75
        self.default_wind_direction = 0.0
        self.default_wind_speed = 3.0
        self.default_fetch = 0.3
        self.default_gamma = 2.0
        self.default_water_depth = 5.0

        self.default_enable_clear_water = False

        # # Surface vessel defaults
        # self.default_starboard_prop_xyz = [0.0, 0.0, 0.0]
        # self.default_port_prop_xyz = [0.0, 0.0, 0.0]


    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass


    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            pass


    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass


    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            #self._reset_extension()
            pass


    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        Water.update_waves_from_UI(False,
                                   self.default_wave_scale,
                                   self.default_directional_spreading,
                                   self.default_wind_direction,
                                   self.default_wind_speed,
                                   self.default_fetch,
                                   self.default_gamma,
                                   self.default_water_depth)

        for ui_elem in self.wrapped_ui_elements:
            # Some of the elements in here don't have a built-in cleanup function
            try: ui_elem.cleanup()
            except Exception as e: pass  # no console message bc it doesn't matter


    def build_ui(self):
        """
        This function will be called any time the UI window is closed and reopened.

        If you need help creating UI elements:
        https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.gui.components/docs/index.html
        """
        wave_settings_frame = CollapsableFrame("Water Settings", collapsed=False)
        with wave_settings_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._enable_clear_water_btn = CheckBox("Enable Clear Water", self.default_enable_clear_water, "", self._update_clear_water)
                self.wrapped_ui_elements.append(self._enable_clear_water_btn)

                self._enable_waves_btn = CheckBox("Enable Waves", self.default_enable_waves, "", self._update_water_settings)
                self.wrapped_ui_elements.append(self._enable_waves_btn)

        self.wave_settings_frame_2 = Frame(enabled=False, visible=False)
        with self.wave_settings_frame_2:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # These return: Tuple(AbstractValueModel, IntSlider)
                # https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.AbstractValueModel.html
                self.wave_scale_slider = combo_floatfield_slider_builder(
                    label="Wave Scale",
                    default_val=self.default_wave_scale,
                    min=0.05,
                    max=2.0,
                    step=0.05,
                )
                self.wave_scale_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.wave_scale_slider)

                self.directional_spreading_slider = combo_floatfield_slider_builder(
                    label="Wave Directionality Spreading",
                    default_val=self.default_directional_spreading,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                )
                self.directional_spreading_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.directional_spreading_slider)

                self.wind_direction_slider = combo_floatfield_slider_builder(
                    label="Wind Direction",
                    default_val=self.default_wind_direction,
                    min=0.0,
                    max=3.14,
                    step=0.01,
                )
                self.wind_direction_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.wind_direction_slider)

                self.wind_speed_slider = combo_floatfield_slider_builder(
                    label="Wind Speed (m/s)",
                    default_val=self.default_wind_speed,
                    min=0.1,
                    max=30.0,
                    step=0.1,
                )
                self.wind_speed_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.wind_speed_slider)

                self.fetch_slider = combo_floatfield_slider_builder(
                    label="Fetch (km)",
                    default_val=self.default_fetch,
                    min=0.1,
                    max=1000.0,
                    step=0.1,
                )
                self.fetch_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.fetch_slider)

                self.gamma_slider = combo_floatfield_slider_builder(
                    label="JONSWAP Gamma",
                    default_val=self.default_gamma,
                    min=0.5,
                    max=10.0,
                    step=0.1,
                )
                self.gamma_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.gamma_slider)

                self.water_depth_slider = combo_floatfield_slider_builder(
                    label="Water Depth (m)",
                    default_val=self.default_water_depth,
                    min=1.0,
                    max=4000.0,
                    step=1.0,
                )
                self.water_depth_slider[0].add_value_changed_fn(self._update_water_settings)
                self.wrapped_ui_elements.append(self.water_depth_slider)

        # surface_vessel_settings_frame = CollapsableFrame("Surface Vessel Settings", collapsed=False)
        # with surface_vessel_settings_frame:
        #     with ui.VStack(style=get_style(), spacing=5, height=0):
        #         self.starboard_prop_xyz = xyz_builder(
        #             label='Location of Startboard Propeller',
        #             axis_count=3,
        #             default_val=self.default_starboard_prop_xyz,
        #             min=-inf,
        #             max=inf,
        #             step=0.001,
        #             on_value_changed_fn=[None, None, None],
        #         )
        #         self.wrapped_ui_elements.append(self.starboard_prop_xyz)

        #         self.port_prop_xyz = xyz_builder(
        #             label='Location of Port Propeller',
        #             axis_count=3,
        #             default_val=self.default_port_prop_xyz,
        #             min=-inf,
        #             max=inf,
        #             step=0.001,
        #             on_value_changed_fn=[None, None, None],
        #         )
        #         self.wrapped_ui_elements.append(self.port_prop_xyz)

    def _update_clear_water(self, clear):
        # list all attributes
        # for attribute in Water.water_prim.GetAttributes():
        #     print()
        #     print("name:", "{}:{}".format(attribute.GetNamespace(), attribute.GetBaseName()) if attribute.GetNamespace() else attribute.GetBaseName())
        if Water.water_prim is not None:
            # if Water.water_prim.GetAttribute("primvars:doNotCastShadows").Get():
            #     Water.water_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool)
            # Water.water_prim.GetAttribute("primvars:doNotCastShadows").Set(clear)

            pxr_prim = omni.usd.get_context().get_stage().GetPrimAtPath(Sdf.Path(Water.water_path))
            pxr_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(clear)


    def _update_water_settings(self, enable_waves):
        if enable_waves:
            self.wave_settings_frame_2.visible = True
            self.wave_settings_frame_2.enabled = True
        else:
            self.wave_settings_frame_2.visible = False
            self.wave_settings_frame_2.enabled = False

        Water.update_waves_from_UI(enable_waves,
                                   self.wave_scale_slider[0].get_value_as_float(),
                                   self.directional_spreading_slider[0].get_value_as_float(),
                                   self.wind_direction_slider[0].get_value_as_float(),
                                   self.wind_speed_slider[0].get_value_as_float(),
                                   self.fetch_slider[0].get_value_as_float(),
                                   self.gamma_slider[0].get_value_as_float(),
                                   self.water_depth_slider[0].get_value_as_float())
