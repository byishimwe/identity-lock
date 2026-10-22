"""MQTT Publisher for FaceLocking motor tracking."""
from __future__ import annotations

import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class FaceTrackMQTTPublisher:
    """Manages an MQTT connection and publishes face-centering commands or target steps."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        topic: str = "face_track",
        rate_limit_sec: float = 0.06,  # ~16 updates per second
        max_steps_range: int = 400,    # +-400 steps corresponds to +-35 degrees
        deadband_px: int = 25,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.rate_limit_sec = rate_limit_sec
        self.max_steps_range = max_steps_range
        self.deadband_px = deadband_px

        self._connected = False
        self._client: Optional[mqtt.Client] = None
        self._last_publish_time = 0.0
        self._last_published_message: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if mqtt is None:
            print("[MQTT] WARNING: paho-mqtt library is not installed.")
            return False

        try:
            # Use CallbackAPIVersion.VERSION2 if available in paho-mqtt >= 2.0
            if hasattr(mqtt, "CallbackAPIVersion"):
                self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:
                self._client = mqtt.Client()

            def on_connect(client, userdata, flags, rc, properties=None):
                rc_code = getattr(rc, "value", rc)
                if rc_code == 0:
                    self._connected = True
                    print(f"[MQTT] Connected to broker at {self.broker_host}:{self.broker_port}")
                else:
                    self._connected = False
                    print(f"[MQTT] Connection failed with code {rc_code}")

            def on_disconnect(client, userdata, disconnect_flags_or_rc, reason_code=None, properties=None):
                self._connected = False

            self._client.on_connect = on_connect
            self._client.on_disconnect = on_disconnect

            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            # Give short moment for on_connect callback
            time.sleep(0.1)
            return True
        except Exception as error:
            print(f"[MQTT] Could not connect to {self.broker_host}:{self.broker_port}: {error}")
            self._connected = False
            return False

    def publish_face_position(
        self,
        detected: bool,
        dx: float,
        frame_width: int,
        mode: str = "steps",
    ) -> Optional[str]:
        """Publishes motor command or step position to MQTT.

        mode = 'steps': Sends integer target step position (-400 to +400) for pointer.
        mode = 'direction': Sends 'LEFT', 'RIGHT', 'CENTER', or 'NOT_DETECTED'.
        """
        if not self._connected or self._client is None:
            return None

        now = time.time()
        if (now - self._last_publish_time) < self.rate_limit_sec:
            return None

        if not detected:
            message = "NOT_DETECTED"
        elif mode == "direction":
            if dx < -self.deadband_px:
                message = "LEFT"
            elif dx > self.deadband_px:
                message = "RIGHT"
            else:
                message = "CENTER"
        else:
            # Normalized position from -1.0 (left) to +1.0 (right)
            half_w = max(1.0, frame_width / 2.0)
            norm_x = max(-1.0, min(1.0, dx / half_w))
            # If within deadband, snap to exact center 0
            if abs(dx) <= self.deadband_px:
                target_steps = 0
            else:
                target_steps = int(round(norm_x * self.max_steps_range))
            message = str(target_steps)

        # Skip duplicate messages if unchanged (except periodically)
        if message == self._last_published_message and (now - self._last_publish_time) < 0.5:
            return None

        try:
            self._client.publish(self.topic, message, qos=0)
            self._last_publish_time = now
            self._last_published_message = message
            return message
        except Exception as error:
            print(f"[MQTT] Publish error: {error}")
            return None

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False
