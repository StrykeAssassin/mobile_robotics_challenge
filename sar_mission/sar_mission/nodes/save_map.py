"""save_map: writes an OccupancyGrid topic to <name>.pgm + <name>.yaml (map_server format) and exits.

  ros2 run sar_mission save_map --ros-args -p topic:=/map -p out:=~/maps/course
  ros2 run sar_mission save_map --ros-args -p topic:=/sar/grid -p out:=~/maps/course_livegrid

Run it while RTAB-Map is still running at the end of the mapping run (before Ctrl+C), so /map is
being published. `ros2 topic list | grep map` shows the topic name if it isn't /map.
Also writes <name>_meta.yaml with the start corner, so the map can be lined up with the arena frame.
"""
import os

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from . import ros_utils as U


class SaveMap(Node):
    def __init__(self):
        super().__init__('save_map')
        self.topic = U.param(self, 'topic', '/map')
        self.out = os.path.expanduser(U.param(self, 'out', '~/maps/course'))
        self.corner = U.param(self, 'start_corner', 'SW')
        self.done = False
        self.create_subscription(OccupancyGrid, self.topic, self.on_map, U.RELIABLE_QOS)
        self.create_timer(30.0, self.timeout)
        self.get_logger().info('waiting for %s ...' % self.topic)

    def on_map(self, msg):
        if self.done:
            return
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        img = np.full(data.shape, 205, dtype=np.uint8)          # unknown = grey
        img[(data >= 0) & (data < 25)] = 254                     # free = white
        img[data >= 65] = 0                                      # occupied = black
        img = img[::-1]                                          # pgm rows go top-down
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        with open(self.out + '.pgm', 'wb') as f:
            f.write(b'P5\n%d %d\n255\n' % (info.width, info.height))
            f.write(img.tobytes())
        o = info.origin.position
        with open(self.out + '.yaml', 'w') as f:
            f.write('image: %s.pgm\nresolution: %.4f\norigin: [%.4f, %.4f, 0.0]\n'
                    'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n'
                    % (os.path.basename(self.out), info.resolution, o.x, o.y))
        with open(self.out + '_meta.yaml', 'w') as f:
            f.write('source_topic: %s\nframe_id: %s\nstart_corner: %s\n'
                    'note: map origin frame = car pose at the start of the mapping run\n'
                    % (self.topic, msg.header.frame_id, self.corner))
        known = np.count_nonzero(data >= 0) * info.resolution ** 2
        self.get_logger().info('saved %s.pgm/.yaml: %dx%d cells at %.2f m, %.0f m^2 mapped'
                               % (self.out, info.width, info.height, info.resolution, known))
        self.done = True
        raise SystemExit

    def timeout(self):
        if not self.done:
            self.get_logger().error('no message on %s after 30 s: is RTAB-Map still running? '
                                    '(ros2 topic list | grep map)' % self.topic)
            raise SystemExit


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = SaveMap()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
