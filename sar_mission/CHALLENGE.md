# The scored run (scoring sheet, 27 Nov 2026)

What the sheet changed, and what the code now does about it.

| Sheet | Before | Now |
|---|---|---|
| Arena about 10 × 7 m, walls, 1.5 × 1.5 m corner zones | 20 × 20 m, 10 × 10 m search area, 3 × 3 m zones | `arena_size: [10.0, 7.0]`, `zone_size: 1.5`; the car starts at the zone centre (`start_inset: 0.75`) facing along the long wall |
| Tmax 10 min; full time points up to 5 min | search to 15:00, exit by 17:00 | first lap to 2:00, search to 7:30 unless all 12 cards are in earlier, forced exit 8:20, stop 9:50 |
| Exit = one of the other three corners, shown by a coloured star on one target | nearest exit matching a star | same, but the start zone is never an exit, and a gate once picked is kept |
| Run ends when the car is fully inside a zone it treats as the exit | – | the driver keeps out of the three other zones while searching (a keep-out line along their inner edges) and opens only the chosen exit zone when leaving |
| One answer file, first 12 entries count, world frame given at the start | `~/sar_targets.csv`, every track, arena frame | `~/sar_run/answer.csv`: `x,y,class`, at most 12 lines, best first, one line per card, in `answer_frame`; cleared when the code starts (a restart clears the answer file) |
| Mapping bonus: map built on board during the run with each target's class and position | – | `~/sar_run/map.pgm/.yaml` (the grid built during the run), `map_targets.json` and `map_semantic.png`, saved every 10 s and at shutdown |

## Searching a 10 × 7 m arena with a 1.65 m turn radius

Lawnmower lanes don't fit: across 7 m there is room for two lanes 3.3 m apart, which is a loop.
So the search is laps (`core/waypoints.ArenaRoutes`):

1. leave the start zone straight ahead, join the **ring** about 3 m out. The ring runs 1.5 m from the
   walls with 2 m radius ends (wider than the turn radius, so a wonky car can still follow it);
2. then alternate a **figure of eight** (two 2 m circles that touch in the middle, one driven each way)
   and the ring. The figure of eight looks at the middle of the arena from every side and turns the
   car round, so the next ring goes the other way: each card gets seen from different directions.

Cards are read from the laps (the recognizer works from about 4 m), with no stops on the way
(`route_inspect: none`). After the laps the car drives to cards it has seen but not read
(`mopup_visits`), then laps again while fewer than 12 are confirmed and there is time.

Leaving: the car stays on the ring until it runs along a wall towards the exit zone, eases over to
the zone's centre line and drives straight in along the wall (`exit_via_route`). A tight turn into a
corner is what got the car stuck in the simulator. If the way is blocked for 20 s it goes round the
other way (`turn_back_on_block`).

Driver changes for the small arena: arcs checked 2.2 m ahead (was 3 m), 0.3 m clearance (was 0.36),
the walls are the fence (was 0.6 m inside a taped line), reversing only along an arc that is clear
behind the car (it used to reverse into walls and stay there), and the last 1.6 m to a goal only
needs that much clear space (or the walls round a corner zone kept the car out).

## The answer file

`target_tracker` rewrites `answer_file` every second (written to a temporary file, then renamed, so the
judges never copy a half-written file):

```
x,y,class
1.92,2.35,HOSTAGE
...
```

Order: confirmed cards (best readings first), then cards read but not confirmed (most likely class),
then cards seen but never read, with a guessed class (the position can still earn 3.33 points).
Never more than 12 lines, never two lines for one card (closer than 0.6 m).

**Frame** (`answer_frame`, ask the TA what "the world frame given at the start" is):
- `start` (default): origin where the car starts, x along its initial heading, y to its left. This is
  the odometry frame, so it needs no measurement.
- `zone`: origin at the outer corner of the start zone, axes along the walls (x along the heading).
- `arena`: origin at the SW corner, x along the 10 m wall.
- `custom`: `answer_origin: [x, y, yaw_deg]`, the judges' frame given in the arena frame.

## The semantic map (mapping bonus)

The map is the occupancy grid `path_planner` builds during the run from the depth scan on the ZED's
pose (visual-inertial odometry with loop closure; corrected by the map localizer when a course map is
loaded). The ZED pose with this live grid is our SLAM. If the TA says only a full SLAM package counts,
run RTAB-Map during the run and set `map_topic: /map`. `target_tracker` saves it to `run_dir` with
the answer entries drawn on it:

- `map.pgm` + `map.yaml`: the grid (map_server format), in the start frame;
- `map_targets.json`: each target with class, position (answer frame and start frame), confidence,
  confirmed or not;
- `map_semantic.png`: picture of the grid, the targets coloured by class with rank and class, and the
  car's track.

## Simulator

```bash
python3 tools/offline_sim.py --seeds 0-15 --reverse                 # the challenge arena is the default
python3 tools/offline_sim.py --seed 2 --reverse --plot run.png --save-run out/   # + answer file and map
python3 tools/offline_sim.py --course old ...                       # the 20 x 20 m arena
```

Layouts follow the sheet: 12 targets (3 per class) at least 1 m from every zone, 3–5 obstacles with
at least 0.8 m clear between any two, INJURED against an obstacle, each THREAT 0.5–0.9 m from a
HOSTAGE, one SAFE card with the star (never the start zone's colour). The score is the sheet's
formula (`core/score.py`), reading the answer list the way the judges would.

Simulator, 16 random layouts (`--seeds 0-15 --reverse`), car that reverses,
1.65 m turn radius, 3-position steering. Score = the sheet's formula including the +5 mapping bonus,
without the algorithm-quality bonus (max 110 here):

| Car | Mean score | Worst | Correct class (of 12) | Position within 0.75 m | Reached the right exit | Mean exit time | Runs with bumps |
|---|---|---|---|---|---|---|---|
| Good steering | **81.1** | 58.7 | 9.7 | 9.4 | 12 / 16 | 7:29 | 3 (one with 100+, capped at −5) |
| Wonky (`--wonky`) | **76.5** | 33.3 | 9.4 | 9.2 | 10 / 16 | 7:25 | 5 (one with 100+) |

Where the points go: about 2.5 cards per run are never read (printed side facing a wall or an
obstacle, or two cards close together counted once); a quarter of the runs don't reach the exit (the
car gets stuck in a pocket between cards and a wall, or never reads the star and guesses the wrong
corner); and leaving at about 7:30 instead of 5:00 costs 5 time points. Tried and dropped: stopping
for cards on the laps, laps in one direction only, turning round when a lap is blocked during the
search (lots of bumps), ending the search at 5:30 (no better: fewer cards).
The simulator's car and camera are models: treat the numbers as a comparison between settings,
not a prediction.

## Questions for the TA

1. The world frame "given at the start": origin and axes (the car's start pose, the start zone's
   corner, or a corner of the court)? Set `answer_frame` to match.
2. Does an occupancy grid built on the ZED's visual SLAM pose count as SLAM for the mapping bonus, or
   must it be a SLAM package (RTAB-Map)?
3. Which way is the court relative to the start (is x along the 10 m wall)? And the heading the car
   is placed at.
4. Can the star be on any target, or only on SAFE ones (we read it on SAFE cards, as on the cards we
   have)?
5. Does driving through a zone that is not the exit end the run, or only stopping in it?
