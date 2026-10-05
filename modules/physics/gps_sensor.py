import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import omni.timeline
import omni.usd
import rclpy
from pxr import Gf, UsdGeom
from sensor_msgs.msg import NavSatFix, NavSatStatus

try:
    from gps_msgs.msg import GPSFix, GPSStatus
except ImportError:
    GPSFix = None
    GPSStatus = None


@dataclass(frozen=True)
class SolutionProfile:
    navsat_status: int
    gpsfix_status: Optional[int]
    horizontal_std_m: float
    vertical_std_m: float
    horizontal_speed_std_mps: float
    vertical_speed_std_mps: float
    bias_walk_std_m_sqrt_hz: float
    hdop: float
    vdop: float
    satellites_used: int
    satellites_visible: int


class GPSSensor:
    EARTH_RADIUS_M = 6378137.0
    MULTI_CONSTELLATION_SERVICE = (
        NavSatStatus.SERVICE_GPS
        | NavSatStatus.SERVICE_GLONASS
        | NavSatStatus.SERVICE_COMPASS
        | NavSatStatus.SERVICE_GALILEO
    )

    _GPSFIX_STATUS_FIX = GPSStatus.STATUS_FIX if GPSStatus is not None else None
    _GPSFIX_STATUS_SBAS = GPSStatus.STATUS_SBAS_FIX if GPSStatus is not None else None
    _GPSFIX_STATUS_DGPS = GPSStatus.STATUS_DGPS_FIX if GPSStatus is not None else None
    _GPSFIX_STATUS_RTK_FIX = GPSStatus.STATUS_RTK_FIX if GPSStatus is not None else None
    _GPSFIX_STATUS_RTK_FLOAT = (
        GPSStatus.STATUS_RTK_FLOAT if GPSStatus is not None else None
    )

    SOLUTION_PROFILES = {"standalone": SolutionProfile(
                            navsat_status=NavSatStatus.STATUS_FIX,
                            gpsfix_status=_GPSFIX_STATUS_FIX,
                            horizontal_std_m=1.2,
                            vertical_std_m=2.0,
                            horizontal_speed_std_mps=0.10,
                            vertical_speed_std_mps=0.15,
                            bias_walk_std_m_sqrt_hz=0.03,
                            hdop=1.1,
                            vdop=1.7,
                            satellites_used=14,
                            satellites_visible=20,),

                        "dgps": SolutionProfile(
                            navsat_status=NavSatStatus.STATUS_GBAS_FIX,
                            gpsfix_status=_GPSFIX_STATUS_DGPS,
                            horizontal_std_m=0.35,
                            vertical_std_m=0.60,
                            horizontal_speed_std_mps=0.05,
                            vertical_speed_std_mps=0.08,
                            bias_walk_std_m_sqrt_hz=0.01,
                            hdop=0.9,
                            vdop=1.4,
                            satellites_used=16,
                            satellites_visible=22,),
                            
                        "sbas": SolutionProfile(
                            navsat_status=NavSatStatus.STATUS_SBAS_FIX,
                            gpsfix_status=_GPSFIX_STATUS_SBAS,
                            horizontal_std_m=0.60,
                            vertical_std_m=1.00,
                            horizontal_speed_std_mps=0.06,
                            vertical_speed_std_mps=0.10,
                            bias_walk_std_m_sqrt_hz=0.015,
                            hdop=1.0,
                            vdop=1.5,
                            satellites_used=15,
                            satellites_visible=21,),

                        "rtk_float": SolutionProfile(
                            navsat_status=NavSatStatus.STATUS_GBAS_FIX,
                            gpsfix_status=_GPSFIX_STATUS_RTK_FLOAT,
                            horizontal_std_m=0.08,
                            vertical_std_m=0.15,
                            horizontal_speed_std_mps=0.03,
                            vertical_speed_std_mps=0.05,
                            bias_walk_std_m_sqrt_hz=0.003,
                            hdop=0.7,
                            vdop=1.0,
                            satellites_used=17,
                            satellites_visible=24,),

                        "rtk_fixed": SolutionProfile(
                            navsat_status=NavSatStatus.STATUS_GBAS_FIX,
                            gpsfix_status=_GPSFIX_STATUS_RTK_FIX,
                            horizontal_std_m=0.015,
                            vertical_std_m=0.03,
                            horizontal_speed_std_mps=0.01,
                            vertical_speed_std_mps=0.02,
                            bias_walk_std_m_sqrt_hz=0.001,
                            hdop=0.6,
                            vdop=0.9,
                            satellites_used=18,
                            satellites_visible=26,),}
    
    MODE_ALIASES = {"fixed": "standalone"}

    def __init__(
        self,
        prim_path: str = "/World/BlueBoat/gps",
        base_link_path: str = "/World/BlueBoat",
        mount_offset_body=(-0.4861, 0.0, 0.158),
        topic_name: str = "/fix",
        navsatfix_topic_name: str = "/navsatfix",
        gpsfix_topic_name: str = "/gpsfix",
        frame_id: str = "gps",
        origin_lat_deg: float = 42.2930,
        origin_lon_deg: float = -83.7162,
        origin_alt_m: float = 0.0,
        update_hz: float = 20.0,
        solution_mode: str = "rtk_fixed",
        publish_gpsfix: bool = True,
    ):
        self.prim_path = prim_path
        self.base_link_path = base_link_path
        self.mount_offset_body = mount_offset_body
        self.topic_name = topic_name
        self.navsatfix_topic_name = navsatfix_topic_name
        self.gpsfix_topic_name = gpsfix_topic_name
        self.frame_id = frame_id

        self.origin_lat_deg = origin_lat_deg
        self.origin_lon_deg = origin_lon_deg
        self.origin_alt_m = origin_alt_m

        self.update_hz = update_hz
        self.publish_gpsfix = publish_gpsfix
        self.solution_mode = self._normalize_solution_mode(solution_mode)

        self._timeline = omni.timeline.get_timeline_interface()
        self._xform_cache = UsdGeom.XformCache()
        self._rng = np.random.default_rng()

        self._origin_world = None
        self._last_world_pos = None
        self._last_track_deg = 0.0
        self._bias_enu_m = np.zeros(3, dtype=float)
        self._accumulated_time = 0.0

        self._ros_initialized_here = False
        self._node = None
        self._navsat_publishers = []
        self._gpsfix_publisher = None
        self._gpsfix_warned = False
        self._prim_fallback_warned = False
        self._ros_init_warned = False

    def _normalize_solution_mode(self, solution_mode: str) -> str:
        normalized_mode = self.MODE_ALIASES.get(solution_mode, solution_mode)
        if normalized_mode not in self.SOLUTION_PROFILES:
            raise ValueError(
                f"Unsupported GPS solution mode '{solution_mode}'. "
                f"Expected one of {sorted(self.SOLUTION_PROFILES)}."
            )
        return normalized_mode

    def set_solution_mode(self, solution_mode: str):
        self.solution_mode = self._normalize_solution_mode(solution_mode)

    def _ensure_ros(self):
        if self._navsat_publishers:
            return True

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._ros_initialized_here = True

            self._node = rclpy.create_node("septentrio_gnss_sim")

            navsat_topics = []
            for topic in (self.topic_name, self.navsatfix_topic_name):
                if topic and topic not in navsat_topics:
                    navsat_topics.append(topic)

            self._navsat_publishers = [
                self._node.create_publisher(NavSatFix, topic, 10)
                for topic in navsat_topics
            ]

            if self.publish_gpsfix and GPSFix is not None and self.gpsfix_topic_name:
                self._gpsfix_publisher = self._node.create_publisher(
                    GPSFix, self.gpsfix_topic_name, 10
                )
            elif self.publish_gpsfix and GPSFix is None and not self._gpsfix_warned:
                print(
                    "[GPSSensor] WARNING: gps_msgs is unavailable, "
                    "skipping /gpsfix publication."
                )
                self._gpsfix_warned = True

            return bool(self._navsat_publishers)
        except Exception as exc:
            if not self._ros_init_warned:
                print(f"[GPSSensor] WARNING: failed to initialize ROS2 GPS publisher: {exc}")
                self._ros_init_warned = True
            self._node = None
            self._navsat_publishers = []
            self._gpsfix_publisher = None
            return False

    def shutdown(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        self._navsat_publishers = []
        self._gpsfix_publisher = None

        if self._ros_initialized_here and rclpy.ok():
            rclpy.shutdown()
            self._ros_initialized_here = False

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._xform_cache.Clear()
            self._origin_world = None
            self._last_world_pos = None
            self._last_track_deg = 0.0
            self._bias_enu_m[:] = 0.0
            self._accumulated_time = 0.0
            self._prim_fallback_warned = False
            self._ros_init_warned = False
            self._ensure_ros()
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._xform_cache.Clear()
            self._origin_world = None
            self._last_world_pos = None
            self._last_track_deg = 0.0
            self._bias_enu_m[:] = 0.0
            self._accumulated_time = 0.0

    def _get_world_position(self, stage):
        prim = stage.GetPrimAtPath(self.prim_path)
        local_point = Gf.Vec3d(0.0, 0.0, 0.0)

        if not prim:
            prim = stage.GetPrimAtPath(self.base_link_path)
            if not prim:
                return None
            if not self._prim_fallback_warned:
                print(
                    f"[GPSSensor] WARNING: prim '{self.prim_path}' not found in "
                    f"stage; falling back to '{self.base_link_path}' with mount "
                    f"offset {tuple(self.mount_offset_body)} (from blueboat.xacro)."
                )
                self._prim_fallback_warned = True
            local_point = Gf.Vec3d(*self.mount_offset_body)

        self._xform_cache.Clear()
        world_from_prim = self._xform_cache.GetLocalToWorldTransform(prim)
        world_pos = world_from_prim.Transform(local_point)
        return np.array(
            [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])],
            dtype=float,
        )

    def _sim_time_s(self):
        return max(float(self._timeline.get_current_time()), 0.0)

    def _sim_time_to_stamp(self):
        sim_time_s = self._sim_time_s()
        sec = int(sim_time_s)
        nanosec = int((sim_time_s - sec) * 1.0e9)
        return sec, nanosec

    def _enu_to_geodetic(self, enu_m):
        lat0_rad = math.radians(self.origin_lat_deg)
        lat_deg = self.origin_lat_deg + math.degrees(enu_m[1] / self.EARTH_RADIUS_M)
        lon_deg = self.origin_lon_deg + math.degrees(
            enu_m[0] / (self.EARTH_RADIUS_M * math.cos(lat0_rad))
        )
        alt_m = self.origin_alt_m + enu_m[2]
        return lat_deg, lon_deg, alt_m

    def _build_header(self):
        sec, nanosec = self._sim_time_to_stamp()
        return sec, nanosec

    def _build_navsatfix_msg(self, lat_deg, lon_deg, alt_m, profile):
        msg = NavSatFix()
        msg.header.frame_id = self.frame_id
        sec, nanosec = self._build_header()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec

        horizontal_var = profile.horizontal_std_m ** 2
        vertical_var = profile.vertical_std_m ** 2

        msg.status.status = profile.navsat_status
        msg.status.service = self.MULTI_CONSTELLATION_SERVICE
        msg.latitude = float(lat_deg)
        msg.longitude = float(lon_deg)
        msg.altitude = float(alt_m)
        msg.position_covariance = [
            horizontal_var,
            0.0,
            0.0,
            0.0,
            horizontal_var,
            0.0,
            0.0,
            0.0,
            vertical_var,
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN
        return msg

    def _build_gpsfix_msg(self, navsat_msg, velocity_enu_mps, profile):
        if self._gpsfix_publisher is None or GPSFix is None or GPSStatus is None:
            return None

        vel_east = velocity_enu_mps[0] + self._rng.normal(
            0.0, profile.horizontal_speed_std_mps
        )
        vel_north = velocity_enu_mps[1] + self._rng.normal(
            0.0, profile.horizontal_speed_std_mps
        )
        vel_up = velocity_enu_mps[2] + self._rng.normal(
            0.0, profile.vertical_speed_std_mps
        )

        horizontal_speed = math.hypot(vel_east, vel_north)
        if horizontal_speed > 1.0e-3:
            self._last_track_deg = (
                math.degrees(math.atan2(vel_east, vel_north)) + 360.0
            ) % 360.0

        hdop = max(profile.hdop + self._rng.normal(0.0, 0.03), 0.5)
        vdop = max(profile.vdop + self._rng.normal(0.0, 0.05), 0.7)
        pdop = math.sqrt(hdop ** 2 + vdop ** 2)
        tdop = max(0.7, 0.8 * hdop)
        gdop = math.sqrt(pdop ** 2 + tdop ** 2)

        horizontal_var = navsat_msg.position_covariance[0]
        vertical_var = navsat_msg.position_covariance[8]
        vertical_sigma = math.sqrt(vertical_var)

        msg = GPSFix()
        msg.header = navsat_msg.header
        msg.status.header = navsat_msg.header
        msg.status.status = profile.gpsfix_status or GPSStatus.STATUS_FIX
        msg.status.position_source = GPSStatus.SOURCE_GPS
        msg.status.motion_source = GPSStatus.SOURCE_POINTS
        msg.status.orientation_source = GPSStatus.SOURCE_NONE
        msg.status.satellites_used = profile.satellites_used
        msg.status.satellites_visible = profile.satellites_visible

        msg.latitude = navsat_msg.latitude
        msg.longitude = navsat_msg.longitude
        msg.altitude = navsat_msg.altitude
        msg.track = self._last_track_deg
        msg.speed = horizontal_speed
        msg.climb = vel_up
        msg.pitch = 0.0
        msg.roll = 0.0
        msg.dip = 0.0
        msg.time = self._sim_time_s()

        msg.gdop = gdop
        msg.pdop = pdop
        msg.hdop = hdop
        msg.vdop = vdop
        msg.tdop = tdop

        msg.err = 2.0 * math.sqrt(2.0 * horizontal_var + vertical_var)
        msg.err_horz = 2.0 * math.sqrt(2.0 * horizontal_var)
        msg.err_vert = 2.0 * vertical_sigma
        msg.err_track = (
            min(180.0, math.degrees(profile.horizontal_speed_std_mps / horizontal_speed))
            if horizontal_speed > 0.1
            else 180.0
        )
        msg.err_speed = 2.0 * math.sqrt(2.0) * profile.horizontal_speed_std_mps
        msg.err_climb = 2.0 * profile.vertical_speed_std_mps
        msg.err_time = 0.02
        msg.err_pitch = 0.0
        msg.err_roll = 0.0
        msg.err_dip = 0.0

        msg.position_covariance = navsat_msg.position_covariance
        msg.position_covariance_type = GPSFix.COVARIANCE_TYPE_KNOWN
        return msg

    def on_physics_step(self, dt: float):
        if not self._navsat_publishers and not self._ensure_ros():
            return

        self._accumulated_time += dt
        publish_period = 1.0 / self.update_hz
        if self._accumulated_time < publish_period:
            return

        publish_dt = self._accumulated_time
        self._accumulated_time = 0.0

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        world_pos = self._get_world_position(stage)
        if world_pos is None:
            return

        if self._origin_world is None:
            self._origin_world = world_pos.copy()

        if self._last_world_pos is None or publish_dt <= 0.0:
            velocity_enu_mps = np.zeros(3, dtype=float)
        else:
            velocity_enu_mps = (world_pos - self._last_world_pos) / publish_dt
        self._last_world_pos = world_pos.copy()

        # Treat world axes as local ENU for this pond-scale simulation.
        enu_m = world_pos - self._origin_world

        profile = self.SOLUTION_PROFILES[self.solution_mode]
        sqrt_dt = math.sqrt(max(publish_dt, 0.0))
        self._bias_enu_m[:2] += self._rng.normal(
            0.0, profile.bias_walk_std_m_sqrt_hz * sqrt_dt, size=2
        )
        self._bias_enu_m[2] += self._rng.normal(
            0.0, profile.bias_walk_std_m_sqrt_hz * sqrt_dt
        )

        noisy_enu_m = enu_m + self._bias_enu_m
        noisy_enu_m[:2] += self._rng.normal(0.0, profile.horizontal_std_m, size=2)
        noisy_enu_m[2] += self._rng.normal(0.0, profile.vertical_std_m)

        lat_deg, lon_deg, alt_m = self._enu_to_geodetic(noisy_enu_m)
        navsat_msg = self._build_navsatfix_msg(lat_deg, lon_deg, alt_m, profile)

        for publisher in self._navsat_publishers:
            publisher.publish(navsat_msg)

        gpsfix_msg = self._build_gpsfix_msg(navsat_msg, velocity_enu_mps, profile)
        if gpsfix_msg is not None:
            self._gpsfix_publisher.publish(gpsfix_msg)
