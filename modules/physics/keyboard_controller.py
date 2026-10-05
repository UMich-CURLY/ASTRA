import carb
import numpy as np
import omni.appwindow

class KeyboardController:
    def __init__(self, thrust_step: float = 0.05):
        self.thrust_step = thrust_step
        self._port: float = 0.0
        self._stbd: float = 0.0

        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)


    def get_thruster_commands(self):
        return self._port, self._stbd


    def _on_keyboard_event(self, event: carb.input.KeyboardEvent):
        event_input = event.input
        event_type = event.type

        modifier = 0.0
        if event_type == carb.input.KeyboardEventType.KEY_PRESS:
            modifier = self.thrust_step
        elif event_type == carb.input.KeyboardEventType.KEY_RELEASE:
            modifier = -self.thrust_step
        else:
            return

        if event_input == carb.input.KeyboardInput.Q:
            self._port += modifier
        elif event_input == carb.input.KeyboardInput.A:
            self._port -= modifier
        elif event_input == carb.input.KeyboardInput.U:
            self._stbd += modifier
        elif event_input == carb.input.KeyboardInput.J:
            self._stbd -= modifier

        self._port = np.clip(self._port, -1.0, 1.0)
        self._stbd = np.clip(self._stbd, -1.0, 1.0)


    def reset(self):
        self._port = 0.0
        self._stbd = 0.0


    def shutdown(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None
