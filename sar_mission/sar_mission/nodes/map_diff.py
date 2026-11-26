"""map_diff: card candidates from 'what the camera sees that isn't in the saved map' (known-map mode).

Subscribes: /sar/grid (live OccupancyGrid from path_planner, odom/sar_map frame), odom
Publishes:  /sar/detections (String JSON, same format as board_detector, frame = odom frame):
            kind 'board', position only (no class, no facing), source 'map_diff'.
            target_tracker turns them into targets to inspect; the camera classifier then reads them.
            /sar/map_diff (String JSON, for debugging: candidates and other new objects)

See core/map_diff.py for the rules (tolerance for the rebuilt course, card-sized blobs only, seen in
several updates before being reported).
"""
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import String

from ..core.map_diff import MapDiffParams, MapDiffTracker, find_new_objects
from ..core.perception import encode_detections
from . import ros_utils as U


class MapDiff(Node):
    def __init__(self):
        super().__init__('map_diff')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.arena = U.arena_from_params(self)
        self.prior = U.prior_from_params(self, self.arena)
        d = MapDiffParams()
        names = ['tol', 'min_length', 'max_length', 'max_width', 'max_range']
        self.params = MapDiffParams(**dict((n, float(U.param(self, n, getattr(d, n)))) for n in names))
        self.tracker = MapDiffTracker(min_hits=int(U.param(self, 'min_hits', 3)))
        self.pose = None
        self.grid = None
        self.t0 = U.now_sec(self)
        self.pub = self.create_publisher(String, '/sar/detections', U.RELIABLE_QOS)
        self.dbg = self.create_publisher(String, '/sar/map_diff', U.RELIABLE_QOS)
        self.create_subscription(OccupancyGrid, '/sar/grid', self.on_grid, U.RELIABLE_QOS)
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_timer(float(U.param(self, 'period', 1.0)), self.tick)
        if self.prior is None:
            self.get_logger().error('no prior_map: map_diff does nothing')
        else:
            self.get_logger().info('map_diff ready (tol %.2f m)' % self.params.tol)

    def on_odom(self, msg):
        self.pose, _ = U.pose_from_odom(msg)

    def on_grid(self, msg):
        self.grid = msg

    def tick(self):
        if self.prior is None or self.grid is None or self.pose is None:
            return
        info = self.grid.info
        occ = np.asarray(self.grid.data, dtype=np.int16).reshape(info.height, info.width) >= 65
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y

        def cell_xy(r, c):
            return ox + (np.asarray(c) + 0.5) * res, oy + (np.asarray(r) + 0.5) * res

        robot = (self.pose.x, self.pose.y)
        cands, others = find_new_objects(occ, cell_xy, self.prior, self.arena, robot, self.params)
        now = U.now_sec(self) - self.t0
        steady = self.tracker.update(now, cands)
        if steady:
            items = [{'kind': 'board', 'xyz': [it['x'], it['y'], 0.2], 'conf': 0.3, 'source': 'map_diff'}
                     for it in steady]
            self.pub.publish(String(data=encode_detections(U.now_sec(self), self.odom_frame, items)))
        self.dbg.publish(U.json_msg({'candidates': [dict((k, round(v, 2) if isinstance(v, float) else v)
                                                         for k, v in it.items()) for it in steady],
                                     'new_objects': others[:20]}))


def main(args=None):
    U.spin(MapDiff, args)


if __name__ == '__main__':
    main()
