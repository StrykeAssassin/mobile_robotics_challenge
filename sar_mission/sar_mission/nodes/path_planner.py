"""path_planner: live occupancy grid from the (pseudo-)scan + A* around obstacles.

Subscribes: scan (LaserScan), odom (Odometry), /sar/route (Path from mission_manager)
Publishes:  /sar/path (Path for local_driver), /sar/grid (OccupancyGrid, for rviz and mission_manager)
"""
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import tf2_ros

from ..core.grid import LiveGrid
from ..core.planner import RoutePlanner
from ..core.scan_utils import Scan
from . import ros_utils as U


class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.sensor_frame = U.param(self, 'scan_frame', '')          # '' = use the scan header
        self.sensor_offset = float(U.param(self, 'sensor_offset', 0.2))
        res = float(U.param(self, 'grid_resolution', 0.1))
        size = float(U.param(self, 'grid_size', 46.0))
        self.grid = LiveGrid(-size / 2, -size / 2, size, res)
        self.max_range = float(U.param(self, 'grid_max_range', 5.0))
        self.planner = RoutePlanner(self.grid, coarse_factor=int(U.param(self, 'plan_coarse_factor', 2)),
                                    inflation=float(U.param(self, 'inflation', 0.4)))
        self.pose = None
        self.route = []
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.last_grid_update = 0.0
        self.create_subscription(LaserScan, U.param(self, 'scan_topic', '/sar/scan'), self.on_scan, U.SENSOR_QOS)
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_subscription(Path, '/sar/route', self.on_route, U.RELIABLE_QOS)
        self.path_pub = self.create_publisher(Path, '/sar/path', U.RELIABLE_QOS)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/sar/grid', U.RELIABLE_QOS)
        self.create_timer(float(U.param(self, 'replan_period', 1.0)), self.replan)
        self.create_timer(1.0, self.publish_grid)
        self.get_logger().info('path_planner ready')

    def on_odom(self, msg):
        self.pose, _ = U.pose_from_odom(msg)

    def on_scan(self, msg):
        if self.pose is None:
            return
        t = U.now_sec(self)
        if t - self.last_grid_update < 0.18:          # ~5 Hz is plenty for the grid
            return
        self.last_grid_update = t
        frame = self.sensor_frame or msg.header.frame_id
        sp = U.sensor_pose(self.tf_buffer, self.odom_frame, frame, self.pose, self.sensor_offset)
        self.grid.update(sp, Scan.from_msg(msg), self.max_range)

    def on_route(self, msg):
        route = U.path_points(msg)
        if route != self.route:
            self.route = route
            self.replan()

    def replan(self):
        if self.pose is None:
            return
        pts = self.planner.plan(self.pose, self.route)
        self.path_pub.publish(U.path_msg(pts, self.odom_frame, self.get_clock().now().to_msg()))

    def publish_grid(self):
        g = OccupancyGrid()
        g.header.frame_id = self.odom_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.resolution = float(self.grid.res)
        g.info.width = g.info.height = self.grid.n
        g.info.origin.position.x = self.grid.ox
        g.info.origin.position.y = self.grid.oy
        g.info.origin.orientation.w = 1.0
        p = 1.0 - 1.0 / (1.0 + np.exp(self.grid.L))
        data = np.where(self.grid.L == 0.0, -1, (p * 100).astype(np.int8))
        g.data = data.astype(np.int8).ravel().tolist()
        self.grid_pub.publish(g)


def main(args=None):
    U.spin(PathPlanner, args)


if __name__ == '__main__':
    main()
