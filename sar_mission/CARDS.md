# Card recognition

`board_detector` decides what each card is (SAFE / INJURED / HOSTAGE / THREAT, or the blank back
of a card) and, for SAFE cards, the star colour that says which exits may be used. No training is
needed: it compares what the camera sees with photos of the real cards in `cards/`.

## How it works

| Step | What | Where |
|---|---|---|
| Find | Whole frame: white card outlines (several brightness levels, local contrast, edges). | `CardRecognizer.find_quads` |
| Find (no outline) | If the outline search finds nothing, the most card-like patch of printed ink (coloured or dark, on a pale surround) is searched as a region, e.g. a white card leaning on a white wall. Needs a clearer fit (score 0.70). | `CardRecognizer.colour_regions` |
| Find (far) | Regions where a card should be: things standing on the floor in the depth image (`/sar/floor` plane from `depth_scan`), and cards the tracker already knows about projected into the image (`/sar/targets` candidates, e.g. from `map_diff`). | `board_detector` (`depth_proposals`, `look_at_targets`) |
| Find (YOLO) | Detector boxes, once a model is trained (`tools/TRAINING.md`). | `board_detector` |
| Classify | Straighten the card to 48x64 px and correlate it with every reference (colour + fine detail), after nudging the corners to fit. Below `min_score` (0.60) it's not a card. | `CardRecognizer.classify_region` |
| Check | Big cards (≥ 120 px): the ORB feature matcher double-checks. | `core/card_matcher.py` |
| Star | Top-right box of the straightened card, white-balanced on its own paper; hue picks red / yellow / green / blue, else `none`. | `star_from_card` |
| Decide | The tracker votes over many frames before confirming a card (3+ readings, 70% combined). | `core/tracker.py` |
| Recheck | A confirmed card whose readings averaged under 80% (or a SAFE card with an unsure star) gets one more look from a new direction, if time allows. | `core/mission.py` |

**Confidence values.** Each reading's class probabilities come from the template scores with
`temperature` 9: a clear match gives about 0.9-0.97, a narrow or weak one 0.6-0.85 (about 1 reading
in 7 on the synthetic test). The mission's recheck rule uses the average of these per reading, not the
combined value, which reaches 99% after a few agreeing readings even when each was only 60% sure.

## Synthetic test results

`tools/card_bench` renders ZED2-like 1280x720 frames: cards 0.7–4.5 m away (≈ 60–450 px tall),
turned up to ±60°, leaning back up to 20°, with lighting colour, shading, blur, noise, JPEG and
partial occlusion, plus distractors (blank card backs, boxes, cones, white panels) and 150 frames
with no card at all. Card textures are cut from the original photos, not the reference crops.

720 cards in 600 frames, plus 150 frames with no card ("right class" by card height in pixels):

| Method | < 70 px | 70–100 | 100–140 | 140–200 | ≥ 200 | All right | All wrong | False alarms (empty frames) | Star right |
|---|---|---|---|---|---|---|---|---|---|
| Old ORB matcher, whole frame | 0% | 0% | 20% | 61% | 71% | 30% | 0% | 0 / 150 | 76/77 |
| Old ORB matcher, on the card's box | 0% | 0% | 20% | 59% | 71% | 30% | 0% | 0 / 150 | 76/77 |
| **Recognizer**, whole frame | 44% | 47% | 46% | 75% | 75% | 57% | 1% | 0 / 150 | 150/168 |
| **Recognizer**, whole frame + depth proposals | 72% | 76% | 75% | 92% | 90% | **81%** | 1% | 1 / 150 | 213/238 |
| **Recognizer**, region from a detector box | 78% | 85% | 84% | 92% | 91% | **86%** | 2% | 0 / 150 | 233/258 |
| **Recognizer**, loose region from a tracked card | 80% | 85% | 82% | 94% | 91% | **87%** | 1% | 0 / 150 | 230/256 |

Time per frame on a 2-core x86 laptop CPU (median): whole frame ~130 ms, each region ~120 ms.
The Jetson's ARM cores are slower, so expect ~0.3–0.6 s per frame with two region searches; it
runs in its own node, so it only limits how often cards are classified, not the driving.

"Wrong" counts a card given a different target class (or called a back). False alarms are
detections where there is no card. A miss is safer than a wrong class: the tracker needs several
agreeing frames, and the mission comes back to inspect unconfirmed cards from closer.

**Caveat:** these are synthetic frames made from your 12 phone photos. The real ZED image will
differ (exposure, colour, motion blur, how the cards are mounted), so treat the numbers as a
comparison between methods, not a promise. The next step is a recorded ZED test (below).

## Real ZED test (27 Nov)

Two ZED recordings from the car's height, one frame per second (left camera), each card approached
from far away, passed at an angle, moved by hand and seen up close, mostly against a white wall:

| Recording | Card found | Wrong class | Star colour (SAFE) |
|---|---|---|---|
| THREAT card (`hostage1`), 152 frames | 119 → **133** | 0 | — |
| SAFE green star, 136 frames | 95 → **101** | 0 | wrong 2 → **0**; not read 26 → 21 |

Left: before the real-data fixes; right: after. Remaining misses are mostly the card far away
(~45 px), cut off by the frame edge, edge-on, lying flat or hidden behind a hand or a leg.
Fixes made from these frames: the ink-patch search above; the star is read from a slightly enlarged
outline (on the real cards the star sits right on the corner); green's hue was moved to where the
ZED shows it (paler and more blue-green than the phone photos); the figure's arm can no longer be
read as a red star. Synthetic bench after the change: whole frame 57% → 66%, 0 false alarms.

## Reference photos (`cards/`)

One file per card: `CLASS[_STARCOLOUR]_anything.png`, e.g. `SAFE_blue_01.png`,
`SAFE_none_02.png`, `HOSTAGE_cartoon.png`. Each is the card **board** cropped to its four corners
and straightened (300x400 is fine). All SAFE cards share one design, so the star colour is always
read from the camera image, never taken from the file name.

To add or replace cards, put the photo's card corners into a straightening script, or use
`tools/card_bench` style code: `cv2.getPerspectiveTransform(corners, [[0,0],[300,0],[300,400],[0,400]])`.
Make sure the crop contains only the card (no floor or cardboard around it): background in a
reference lowers the score of the real card and can fake a star colour.

### Photos that would help most

1. **From the ZED itself**, not a phone: run `ros2 run sar_mission dataset_capture` (or record a
   bag of `/zed/zed_node/rgb/image_rect_color/compressed`) while the car looks at each card from
   1, 2, 3 and 4 m, straight on and turned ~45°. This is the real test set.
2. **Every card design that exists.** If the organisers have more designs than the 12 photographed
   (other HOSTAGE / INJURED / THREAT pictures), each one needs a reference photo; an unknown design
   can only come out as "not a card".
3. **The back of a card** and how cards are mounted (leaning on a box? on a stand?), so the back
   check and the outline finder can be tuned.
4. The **lighting on site** (one ZED frame of the arena is enough to check colours and exposure).

## Tuning (board_detector parameters, `config/car.yaml`)

| Parameter | Default | Effect |
|---|---|---|
| `use_recognizer` | true | false = old ORB-only matcher |
| `min_score` | 0.60 | raise (0.65) if you ever see a wrong class; lower (0.55) if real cards are missed |
| `matcher_rate` | 2.0 | whole-frame searches per second without YOLO |
| `look_at_targets`, `depth_proposals` | true | region searches for far cards |
| `look_max` | 2 | regions searched per frame per source (each ~0.1–0.3 s on the Jetson CPU) |
| `card_height`, `camera_height` | 0.42, 0.25 | MEASURE: board height and lens height (m), used to size the regions |

## Re-running the test

```bash
cd tools/card_bench
python3 gen.py scenes                      # 600 scenes + 150 empty ones (~3 min)
python3 evaluate.py new_region scenes 250  # or: new_box new_full new_depth new_look orb_box orb_full
```
`gen.py --photos photos.json` cuts the test cards from your original photos (better than the
default, which uses the reference crops). `CARD_REFS=/path/to/cards python3 evaluate.py ...` tests
another reference folder.
