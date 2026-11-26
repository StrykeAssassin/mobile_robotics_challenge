# Mapping session guide: auto-mapping the course with RTAB-Map

**Goal:** one detailed RTAB-Map map of the fixed course, the 4 start positions and the exit points marked in it, and a small rosbag for testing at home.

**Time:** about 1.5 h. That's ~20 min setup, ~15 min of car measurements, ~25 min for the auto-mapping run, and ~20 min for saving and checking.

## 0. Before the session (at home)

1. Copy `sar_mission` onto the car and build it:
   ```bash
   scp -r sar_mission nvidia@<car-ip>:~/ros2_ws/src/
   ssh nvidia@<car-ip>
   cd ~/ros2_ws && colcon build --packages-select sar_mission && source install/setup.bash
   ros2 pkg list | grep -E "zed_wrapper|rtabmap"      # both must be there
   mkdir -p ~/maps ~/bags && df -h ~                  # keep at least 5 GB free
   ```
2. On your laptop, use the **same** `ROS_DOMAIN_ID` as the car and pick a number no other team uses, e.g. `export ROS_DOMAIN_ID=8` on both. Check that `ros2 topic list` on the laptop shows the car's topics.

## 1. Prepare the course (≈ 20 min)

1. **Remove every cardboard card from the course.** The cards move on the day. If they're in the map, the car will avoid ghosts and think the floor around them is blocked. Leave everything that stays fixed: boxes, cones, crates, mats and exits.
2. **Define the arena frame.**
   - Pick one corner as **SW = (0, 0)**.
   - **x** runs along one wall to the next corner (SE).
   - **y** runs along the other wall (to NW).
   - Write down which physical corner is which and take a photo. These names are what `start_corner:=` means.
3. **Tape the 4 start positions** at the centre of each start zone. Add a short line showing the car's heading at each one, squared to the wall: SW faces +x (towards SE), SE faces +y, NE faces −x, NW faces −y. If the official start marks differ, tape those and measure them instead.
4. **Measure with a tape** from the SW corner, in metres:
   - each start mark (x, y);
   - each exit point: its colour, centre (x, y) and size;
   - 4–5 obstacle corners spread around the course, to check the map later.

   Write them all into `course_measurements.yaml`; a template is at the end of this guide.

## 2. Measure the car (≈ 15 min, do this first)

Do these with the remote and a tape measure. Put the results in `config/mapping.yaml` **and** `config/car.yaml`.

| Test | How | Parameter |
|---|---|---|
| Turn radius | On slow, full lock, drive one full circle **left and one right**. Measure each diameter and halve it. Do each twice; the spread tells you how wonky it is. | `turn_radius` (average), plus `turn_radius_left` / `turn_radius_right` if they differ by more than ~15% |
| Steering type | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.05}}"`, then again with `z: 0.15`. Same wheel angle both times? | same angle → `steering_mode: three_value`, different → `proportional` |
| Reverse | `ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.3}}"` with the wheels off the ground. Does it reverse? | `reverse_allowed` |
| Slow speed | Time 5 m on slow. | for planning time |
| Camera height | Floor to the middle of the ZED lens. | `camera_height` |

**If you have no time:** set `turn_radius: 1.9`. Too big is safe (the car just takes wider turns); too small makes it attempt turns it can't make.

**The car is wonky, and the mapper is built for that:**
- A turn only counts as safe if it's clear even when the real turn comes out 25% tighter **or** 25% wider (`radius_spread`), and when "straight" drifts slightly.
- Steering changes at most every 0.5 s (`steer_hold`), so the servo doesn't chatter.
- The car **stops before every steering change** (`steer_change_pause`), so each turn starts from a standstill, and also stops every 1.5 m to let RTAB-Map catch up.
- There's no throttle pulsing, so there aren't dozens of lurchy starts a minute.
- It uses 3 lanes only if the U-turn fits with margin, otherwise 2 wider lanes.

In simulation the car got random turn radii (−20% to +25% on every turn), a 12% left/right difference, steering lag, speed wobble and start lurches. It still mapped 99% of the arena with zero bumps from all four corners. That held even when its real turn radius (1.95 m) was wider than the one it was told (1.65 m).

**If it's still too lively:** increase `steer_change_pause` and `pause_time`, or reduce `pause_every`. If it hesitates too much in tight spots, lower `radius_spread` to 0.15.

## 3. The auto-mapping run (≈ 25 min)

On the car, in separate terminals (`source ~/ros2_ws/install/setup.bash` in each):

```bash
remote_control                                     # then reset the Arduino
ros2 launch car_control car_control.launch.py
ros2 launch sar_mission mapping.launch.py start_corner:=SW
```

`mapping.launch.py` starts the ZED, RTAB-Map and `depth_scan`. RTAB-Map gets Tutorial 4's arguments plus two extras: depth is only trusted up to 4 m, and free space is ray-traced. Add `rtabmap_extra:=""` to get exactly the tutorial's settings. After 15 s it starts `auto_mapper`. It also records a small rosbag to `~/bags/mapping_run`.

Put the car on the SW tape mark, squared to the wall, **before** launching. Then switch the remote to autonomous control.

On the laptop, watch:

```bash
rviz2 -d ~/sar_ws/src/sar_mission/rviz/sar.rviz        # add a Map display for /map as well
ros2 topic echo /sar/mapper_status
```

**What the car does:**
1. Two laps of the corridor around the search area. Two laps give RTAB-Map its loop closures.
2. Three lanes east–west, then three north–south, so every obstacle is seen from several sides.
3. One lap the other way round.
4. Any gaps that are still unmapped.

It drives on the slow setting and stops for 1.5 s every 1.5 m, and briefly before every steering change. That's the "move and pause" advice from Tutorial 4. In simulation, with the wonky car model, the whole run took about 30 min and 240–280 m.

**Pause or resume from the laptop** (e.g. if people walk through the course):

```bash
ros2 topic pub --once /sar/mapper_enable std_msgs/msg/Bool "{data: false}"   # pause
ros2 topic pub --once /sar/mapper_enable std_msgs/msg/Bool "{data: true}"    # resume
```

**Safety:**
- One person always holds the remote, ready to take over. The node also stops by itself if odometry or the depth scan stops, and holds still for 4 s if the odometry jumps (tracking lost).
- Keep people out of the course during the run. They'd end up in the map.
- If RTAB-Map loses odometry often, pause, take over, drive back a little by hand, then resume.

**Fallback:** if the auto mapper can't get through somewhere tight, run `mapping.launch.py drive:=false`. Everything except the driving starts, and you drive the same pattern by hand: two corridor laps, criss-cross lanes, one lap the other way.

## 4. Save the map (before closing anything!)

When the log says **MAPPING RUN FINISHED**, keep RTAB-Map running and in a new terminal:

```bash
ros2 run sar_mission save_map --ros-args -p topic:=/map -p out:=~/maps/course -p start_corner:=SW
ros2 run sar_mission save_map --ros-args -p topic:=/sar/grid -p out:=~/maps/course_livegrid
```

If `/map` isn't found, check `ros2 topic list | grep map` and use the topic it shows.

Then Ctrl+C the mapping launch. RTAB-Map writes `~/.ros/rtabmap.db`. **Copy it immediately**, because the next RTAB-Map start deletes it:

```bash
cp ~/.ros/rtabmap.db ~/maps/course_$(date +%m%d_%H%M).db
```

## 5. Check the map (≈ 10 min)

```bash
rtabmap-databaseViewer ~/maps/course_<date>.db
```

Tutorial 4, sections 1.3 and 2.1, show how to open the graph view and edit the 2D map. Check that:
- The walls are straight and the corners square. Bent walls mean drift; doubled walls mean a missed loop closure.
- The obstacle positions match your tape measurements to within about 10–15 cm.
- There are no ghost obstacles where people stood. Remove them with **Edit optimized 2D map → Clear obstacle**, as in Tutorial 4 section 1.3.

**Good enough means:**
- walls straight and corners square;
- obstacles within about 10 cm of your tape measurements;
- no doubled walls and no ghost obstacles;
- every taped start mark and exit point visible in the map.

If it meets these, stop mapping and spend the rest of the session photographing the cards. If it doesn't, do another run from a different corner and keep the better map.

## 6. Bring back

- `~/maps/course_<date>.db`
- `~/maps/course.pgm` + `.yaml` + `_meta.yaml`, and the `course_livegrid` files
- `~/bags/mapping_run/`
- `course_measurements.yaml` (below), plus photos of the course from each corner

These feed known-map mode (**KNOWN_MAP.md**): convert the iPad scan with `tools/ply_to_map.py`,
compare it with this map using `tools/compare_maps.py --merged`, and launch the mission with
`prior_map:=` and `course_file:=`.

## `course_measurements.yaml` template

```yaml
# Arena frame: origin at the corner you called SW, x towards SE, y towards NW, metres.
arena_size: [20.0, 20.0]          # measure: the real course may not be exactly 20 x 20
corners_photo: "IMG_....jpg"      # photo showing which corner is SW
start_positions:                  # tape mark centres, heading in degrees (0 = +x, 90 = +y)
  SW: {x: 1.5, y: 1.5, heading: 0}
  SE: {x: 18.5, y: 1.5, heading: 90}
  NE: {x: 18.5, y: 18.5, heading: 180}
  NW: {x: 1.5, y: 18.5, heading: -90}
exits:                            # one entry per exit point
  - {colour: red,    x: 0.0, y: 0.0, size_x: 0.0, size_y: 0.0}
  - {colour: green,  x: 0.0, y: 0.0, size_x: 0.0, size_y: 0.0}
  - {colour: blue,   x: 0.0, y: 0.0, size_x: 0.0, size_y: 0.0}
  - {colour: yellow, x: 0.0, y: 0.0, size_x: 0.0, size_y: 0.0}
check_points:                     # obstacle corners measured by tape, to check the map
  - {name: "crate A, corner nearest SW", x: 0.0, y: 0.0}
car:
  turn_radius: 0.0
  steering_mode: three_value      # or proportional
  reverse: false
  slow_speed_mps: 0.0
  camera_height: 0.0
```
