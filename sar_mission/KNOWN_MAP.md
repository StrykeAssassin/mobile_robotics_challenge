# Known-map mode: using the saved course map on challenge day

The course layout is fixed; only the cards change, and the course is rebuilt every day so pieces may
sit 10–30 cm from where they were mapped. Known-map mode uses the saved map as a **head start, not
the truth**:

| Part | What it does | File |
|---|---|---|
| Prior map | Course map in the arena frame (ZED, iPad or merged). | `core/prior_map.py` |
| Live grid + prior | The planner's grid starts from the map. Mapped obstacles clear quickly when the camera sees the spot empty; what the camera sees is sticky. | `core/grid.py` (`seed_prior`) |
| Localizer | Particle filter: depth scan vs map (as in HW4). Pulls ZED drift back. Beams that hit unmapped things (cards, moved pieces) count as "unexplained" instead of dragging the pose. Holds the correction when the scan doesn't match well. | `core/localizer.py`, node `localizer` |
| Map difference | Anything the camera sees > 0.45 m from every mapped obstacle, card-sized, and seen in 3 updates becomes a target to inspect. Moved boxes don't trigger it. | `core/map_diff.py`, node `map_diff` |
| Routes | Lanes and ring with every point pushed out of mapped obstacles, plus extra viewpoints where the map says the route can't see. | `core/known_route.py` |
| Card recognizer | Template matching against photos of the real cards (ORB double-checks big ones). Class, and for SAFE cards the star colour read from the image. Works without YOLO, from ~4 m; map_diff targets are projected into the image and classified there. See **CARDS.md**. | `core/card_recognizer.py`, in `board_detector` |
| Exit rule | Nearest exit whose colour matches a star on a SAFE card found. | `core/mission.py`, `core/tracker.py` |
| Measured course | Start marks and exits from `course_measurements.yaml`. | `core/course.py` |

Without `prior_map`, everything runs exactly as before (explore from scratch).

## 1. Make the map (once, then re-check after rebuilds)

```bash
# a) ZED map: the auto-mapper run (MAPPING.md), then while RTAB-Map still runs:
ros2 run sar_mission save_map --ros-args -p topic:=/map -p out:=~/maps/course_day1 -p start_corner:=SW

# b) iPad scan (Dot3D / Modelar ...), exported as PLY, converted to the arena frame:
python3 tools/ply_to_map.py scan.ply --out ~/maps/ipad_day1 --markers markers.yaml --preview ipad.png
#    no markers picked? align it to the ZED map instead:
python3 tools/ply_to_map.py scan.ply --out ~/maps/ipad_day1 --align-to ~/maps/course_day1.yaml

# c) compare them, see where they disagree, and write the map the robot will use:
python3 tools/compare_maps.py ~/maps/ipad_day1.yaml ~/maps/course_day1.yaml --refine \
    --course course_measurements.yaml --png compare.png --merged ~/maps/merged_day1
```

`compare_maps.py` lists every disagreement bigger than 15 cm with arena coordinates. Measure those
with a tape. The same tool compares two days' maps (`course_day1.yaml course_day2.yaml --refine`):
that tells you how much the rebuild really moves things.

**Which map to give the robot:** the merged one if the iPad scan aligned well (most detail),
otherwise the ZED map alone works too (`prior_map:=~/maps/course_day1.yaml`; it is moved into the
arena frame using the start corner saved in its `_meta.yaml`).

## 2. Card photos

`cards/` ships the 12 cards photographed so far (each cropped to the card board, upright). Name
files `CLASS[_STARCOLOUR]_anything.png`, e.g. `SAFE_blue_01.png`, `SAFE_none_02.png`,
`HOSTAGE_cartoon.png`. To use your own folder: `card_refs:=~/cards` (board_detector parameter).
All SAFE cards share one design, so the star colour is always **read from the camera image**
(top-right of the card), never taken from the reference.

The recognizer classifies cards down to ~50-60 px tall (4-5 m at 720p). Without a trained YOLO
model it searches whole frames at 2 Hz, plus regions from the depth image and the cards map_diff
reported (projected into the image). Details, test results and which photos to take: **CARDS.md**.

## 3. Challenge day

```bash
ros2 launch sar_mission car.launch.py start_corner:=NE \
    prior_map:=~/maps/merged_day1.yaml course_file:=~/maps/course_measurements.yaml
```

This adds `localizer` and `map_diff`, and switches every node to the corrected frame `sar_map` /
topic `/sar/odom`. Before pressing O:

```bash
ros2 topic echo /sar/loc_status     # state: tracking, explained above ~0.5, spread below ~0.2
```

In rviz, set the fixed frame to `sar_map`. The live grid (`/sar/grid`) should line up with the walls.
If `explained` is low, the car is probably not on its start mark: re-seat it.

## 4. What happens when the course was rebuilt

| Change | Result |
|---|---|
| Obstacle moved 10–30 cm | The camera sees it in the new place; the old cells clear after a scan or two; the route bends around it. No inspection (within the 0.45 m tolerance). |
| Obstacle missing | Its cells clear; paths get shorter. |
| New obstacle | Card-sized: inspected once (~20 s), then marked not a target. Bigger: just avoided. |
| Lots changed | The localizer's `explained` drops and it **holds** the correction (plain ZED odometry), so a bad map can't drag the pose away. |

Tuning (config/car.yaml): `map_diff/tol` (raise if moved boxes trigger inspections),
`localizer/min_explained`, `localizer/sigma_hit`, `mission_manager/route_clearance`.

## 5. Test it without the car

```bash
python3 -m pytest -q test                                              # 35 tests
python3 tools/offline_sim.py --seeds 0-9 --known-map --shift 0.2       # rebuilt course: obstacles moved up to 20 cm
python3 tools/offline_sim.py --seeds 0-9 --known-map --shift 0.2 --drift 3 --wonky
python3 tools/offline_sim.py --seeds 0-9 --shift 0.2 --drift 3         # same course without the map, to compare
```

`--shift` rebuilds the course (every obstacle moved/turned, cards leaning on them move too);
`--drift` multiplies the odometry error; `--no-localizer` / `--no-mapdiff` switch parts off.

**Results** (offline simulator, 10 random layouts, course rebuilt with obstacles moved up to 20 cm.
Layout 9 happens to have no SAFE card, so no exit can count there: 9 is the maximum):

| Condition | Without the map | With the map |
|---|---|---|
| Odometry drift 3x normal | exit OK 7/10, position error 0.93 m (max 7.7 m), 761 s | **exit OK 9/10**, position error **0.15 m** (max 0.64 m), 558 s |
| Wonky car, no reverse | exit OK 4/10, 3.7 of 8 cards | exit OK 3/10, 4.7 of 8 cards |
| Wonky car, reverse works | exit OK 7/10, 4.8 of 8 cards | **exit OK 8/10, 5.6 of 8 cards**, 616 s vs 626 s |

0 bumps in every run. The localizer keeps the position within ~15 cm where plain odometry
drifted metres. Without reverse, the wonky car still gets boxed in between cards in about half the
runs, with or without the map: that is a driving limit, not a map problem (see Known limits).
The whole node set (localizer, map_diff, planner, driver, tracker, mission) was also run on a
stand-in ROS bus with 3x drift: exited through the allowed gate, position error 0.08 m vs 0.32 m raw.

## 6. Known limits

- **Boxed in, no reverse.** A car that can't reverse can still get trapped between cards and
  obstacles during an inspection. The map doesn't fix that; testing reverse on day 1 does.
- **Cards leaning on obstacles** are within the map tolerance, so map difference can't see them.
  The camera (recognizer / YOLO) finds those.
- **Map difference doesn't find cards by itself:** it only says "look here". The class comes
  from the camera.
- **Routes** are the lanes and ring cleaned against the map, plus up to 6 gap viewpoints. They are
  not a fully optimised viewpoint plan.
- **The iPad `--align-to` search** assumes the ZED map is reasonably good. With a bent ZED map,
  use markers.
