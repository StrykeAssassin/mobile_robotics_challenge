# sar_mission: Group 8 search and rescue stack

One ROS 2 package for the 50.047 search and rescue challenge. It runs:
- on the **RC car** (Jetson, ROS 2 Foxy, ZED2);
- in **Gazebo** on the laptop (WSL2 Ubuntu 22.04, ROS 2 Humble), using a TurtleBot3 restricted to drive like the car;
- in a **plain-Python simulator** that needs no ROS at all (`tools/offline_sim.py`).

The plan is the Phase 1 "Lanes + Ring" strategy:
1. Drive 3 lawnmower lanes through the 10 × 10 m search area.
2. Drive a loop 3 m inside the arena edge.
3. Go back to any board that was seen but not classified (mop-up).
4. Drive to the gate matching the exit sign's colour.

At each board the car stops 1.5 m in front of the printed side and classifies it over several frames. Targets are reported in the arena frame: origin at the SW corner, x east, y north, in metres.

## How it fits together

```
ZED depth ──► depth_scan ──► /sar/scan ─┬─► path_planner ──► /sar/path ──► local_driver ──► /cmd_vel ──► car
                   └─► /sar/floor       │      ▲  └─► /sar/grid             ▲
ZED RGB ──► board_detector ─► /sar/detections ─► target_tracker ─► /sar/targets ─► mission_manager
              (YOLO ONNX)           └──────────► exit_detector ──► /sar/exit, /sar/mats ─┘   │
ZED odom ─────────────────────────────────────────────────────────────────── /sar/route ◄──┘
```

| Node | What it does | Reused from |
|---|---|---|
| `depth_scan` | Fits the floor plane in every depth frame and turns everything 3 cm–0.8 m above it into a LaserScan. Missing depth close in front is treated as blocked. | Camera model from Tutorial 6 |
| `path_planner` | Builds a live log-odds grid from the scan. Plans a straight line if it's clear, otherwise A* with inflation. | HW3 mapper, HW2 A* (bugs fixed) |
| `local_driver` | Pure pursuit + PID. Picks among left / straight / right arcs that stay clear of obstacles and the arena boundary, even if the real turn is 25% tighter or wider than planned. Has watchdog and stuck detection. | HW1 PID, HW3 explore logic |
| `target_tracker` | Places each detection in 3D using depth and TF. Fuses sightings: weighted position, log-probability class votes, and a board-facing estimate. Writes `~/sar_targets.csv`. | HW3 least squares, HW4 weighting |
| `exit_detector` | Exit sign colour vote, plus a map of coloured floor mats (only pixels on the floor count, so cones and the sign are excluded). | — |
| `mission_manager` | State machine with timers. Prints the final report. | — |
| `board_detector` | YOLOv8 detector + crop classifier, run with OpenCV DNN. See `tools/TRAINING.md`. | HW4 detection idea |
| `fake_detector`, `steering_wrapper` | **Simulation only**: stand-in for YOLO, and makes the TurtleBot drive like the car. | — |
| `dataset_capture` | Saves camera frames for training YOLO. | — |
| `auto_mapper` | Drives the course slowly for RTAB-Map. Runs corridor laps, criss-cross lanes and a reverse lap, then fills gaps. See **MAPPING.md**. | HW3 auto_explore idea |
| `save_map` | Saves any OccupancyGrid topic (e.g. RTAB-Map's `/map`) as `.pgm` + `.yaml`. | — |

All the logic is in `sar_mission/core/` (plain Python + numpy, no ROS). The nodes in `sar_mission/nodes/` are thin wrappers, so the same code runs in the unit tests, the offline simulator, Gazebo and on the car.

## 1. Quick check without ROS (any laptop with Python 3)

```bash
cd sar_mission
python3 -m pytest -q test                          # 25 unit tests
python3 tools/offline_sim.py --seed 3 --plot run3.png -v
python3 tools/offline_sim.py --seeds 0-19          # benchmark table
python3 tools/offline_sim.py --seeds 0-9 --turn-radius 1.25 --lanes 4    # "what if" after day-1 tests
python3 tools/offline_sim.py --seeds 0-9 --reverse                        # if the car can reverse
python3 tools/offline_sim.py --seeds 0-9 --wonky                          # badly behaved car
```

Last benchmark (10 random layouts, 3 lanes, 3-position steering, no reverse, turn radius 1.65 m, 1% odometry drift). The **wonky** column uses `--wonky`: random turn radius −20% to +25% per turn, 12% left/right difference, steering lag, speed wobble and start lurches.

| Metric | Ideal car | Wonky car |
|---|---|---|
| Correct exit gate | **10 / 10** | **8 / 10** (2 runs stuck in a cluster of cards: needs reverse) |
| Targets classified correctly | 53 / 74 (72%), 0 false reports | 46 / 74 (62%), 0 false reports |
| Bumps | 0 | 0 |
| Mean run time | 10.1 min | 12.0 min |

The simulator's detector model is an assumption: detect ≤ 5 m, classify ≤ 2.2 m, 15% confusion between similar poses. Real numbers come from the trained model.

## 2. Gazebo (WSL2, ROS 2 Humble)

```bash
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-gazebo-ros-pkgs
mkdir -p ~/sar_ws/src && cp -r sar_mission ~/sar_ws/src/
cd ~/sar_ws && colcon build --symlink-install && source install/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 launch sar_mission sim.launch.py                 # world seed 3, start SW
rviz2 -d ~/sar_ws/src/sar_mission/rviz/sar.rviz        # second terminal
```

Other layouts: `python3 tools/make_world.py --seed 7` (add `--gates midpoints` for one mat per side), then `colcon build` and `ros2 launch sar_mission sim.launch.py seed:=7 start_corner:=NE`.

Use `gui:=false` if Gazebo's window is slow in WSL. The results print in the `mission_manager` output and are saved in `~/sar_targets_sim.csv`.

What Gazebo tests: the planner, driver, mission logic, timers and launch wiring, with real ROS timing. What it doesn't test: perception. `fake_detector` stands in for YOLO and the ZED depth.

## 3. On the car (Foxy)

```bash
# copy the package into the car's workspace, then
cd ~/ros2_ws && colcon build --packages-select sar_mission && source install/setup.bash
```

Bring-up order (from the W3 manual): `remote_control` → reset the Arduino → `ros2 launch car_control car_control.launch.py` → ZED → then:

```bash
ros2 launch sar_mission car.launch.py start_corner:=SW
```

Press **O** on the remote to switch the car to autonomous. The mission clock starts when the car first moves.

The car has `/zed/zed_node/odom`, so no RTAB-Map is needed for this to run. Set a unique `ROS_DOMAIN_ID` on the car and laptop so other teams' commands can't reach your car.

Partial tests:

| Command | Tests |
|---|---|
| `car.launch.py detector:=false mission:=false` | Depth scan only. Watch `/sar/scan` in rviz. |
| `ros2 topic echo /sar/floor` | `pitch_deg` should be steady, `inliers` above 0.5. |
| `ros2 run sar_mission dataset_capture` | Collects YOLO training photos. |

## 3b. Mapping the course (auto mapper)

The course is fixed and only the cards change, so map it in advance. Follow **MAPPING.md**:

```bash
ros2 launch sar_mission mapping.launch.py start_corner:=SW      # ZED + RTAB-Map + depth_scan + auto_mapper
python3 tools/offline_mapper_sim.py --seed 3 --plot mapper.png   # try it offline first
```

## 4. Day-1 measurements → `config/car.yaml`

| Measure | Parameter | Why it matters |
|---|---|---|
| Full-lock turn radius at slow speed | `local_driver/turn_radius` (and `--turn-radius` in the offline sim) | Arc prediction, U-turns, lane spacing (4 lanes only if ≤ 1.25 m) |
| Is steering 3-position or proportional? | `local_driver/steering_mode` | `three_value` or `proportional` |
| Does a negative `linear.x` reverse? | `local_driver/reverse_allowed` | Without reverse, dead ends can cost a bump |
| Real slow speed | `steering_wrapper/sim_slow_speed`, `--speed` | Timing and time budget |
| Camera lens height | `depth_scan/camera_height` | Floor-fit prior |
| ZED objects `header.frame_id` | (only if `use_zed_objects`) | Backup person detection |
| HSV ranges in the hall (lock white balance first) | `exit_detector/s_min`, `v_min`, ranges in `core/colour.py` | Sign and mats |

**Ask the TA:**
- Are the gates corner zones or side mats (`gate_layout`)? What's the colour order (`gate_colours`)?
- What counts as "exited"?
- What does the exit sign look like?
- How should target positions be reported?

## 5. Known limits

- **YOLO isn't trained yet.** Until `~/sar_models/*.onnx` exist, the car searches, avoids obstacles and exits, but logs no targets. The exit sign is read from colour blobs meanwhile.
- **Without reverse,** the car avoids clutter by predicting its arcs and by skipping viewpoints that are cluttered on the map. In a true dead end, after 15 s it makes short moves only along arcs that take it away from obstacles. A wonky car can still get boxed in between cards (2 of 10 simulated runs), which reverse or the planned known-map viewpoints would fix.
- **Positions drift with ZED odometry,** about 1% of distance travelled. The exit gate is refreshed every time its mat is seen again.
- **Not tested on the real car or ZED yet.** Tested so far: 25 unit tests, the offline simulator, and all nodes run together against a stand-in ROS bus. The code avoids Humble-only APIs, but it hasn't been run on Foxy.
