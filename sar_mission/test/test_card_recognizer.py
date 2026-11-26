"""Card recognizer: classification of small / turned cards, star colour, blank backs, region search."""
import glob
import os

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from sar_mission.core.card_recognizer import CardRecognizer, order_corners, star_from_card  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, '..', 'cards')


@pytest.fixture(scope='module')
def rec():
    return CardRecognizer(CARDS)


def scene(card, quad, size=(720, 1280), seed=0, gain=1.0):
    """Paste a card image into a plain floor/wall scene at quad (TL TR BR BL)."""
    rng = np.random.RandomState(seed)
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:h // 2] = (115, 125, 135)
    img[h // 2:] = (95, 95, 97)
    ch, cw = card.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), np.float32(quad))
    warped = cv2.warpPerspective(card, M, (w, h), flags=cv2.INTER_AREA)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(quad), 255)
    img[mask > 0] = warped[mask > 0]
    img = np.clip(img.astype(float) * gain + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def card_quad(cx, bottom, height, turn=0.0):
    """Upright card of `height` px; turn squeezes it sideways (0..0.5) like a card turned away."""
    w = 0.75 * height * (1 - turn)
    d = 0.08 * height * turn
    return np.float32([[cx - w / 2, bottom - height + d], [cx + w / 2, bottom - height],
                       [cx + w / 2, bottom], [cx - w / 2, bottom - d]])


def refs():
    out = []
    for p in sorted(glob.glob(os.path.join(CARDS, '*.png'))):
        name = os.path.basename(p)
        out.append((name, name.split('_')[0], (name.split('_')[1] if name.startswith('SAFE') else None),
                    cv2.imread(p)))
    return out


def test_order_corners():
    q = order_corners([[10, 0], [0, 0], [0, 10], [10, 10]])
    assert q.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


@pytest.mark.parametrize('height,turn', [(220, 0.0), (90, 0.3), (60, 0.0)])
def test_every_card_found_and_classified(rec, height, turn):
    for i, (name, cls, star, img) in enumerate(refs()):
        q = card_quad(640, 420, height, turn)
        found = rec.find_all(scene(img, q, seed=i))
        cards = [f for f in found if not f.back]
        assert cards, '%s at %d px not found' % (name, height)
        assert cards[0].cls == cls, '%s at %d px -> %s' % (name, height, cards[0])
        if cls == 'SAFE':
            assert cards[0].star == (star or 'none'), '%s star -> %s' % (name, cards[0].star)


def test_region_search_without_outline(rec):
    """classify_region finds a card inside a loose region even when the card's edge blends into a
    pale background (no white outline to segment)."""
    name, cls, _, img = [r for r in refs() if r[1] == 'INJURED'][0]
    q = card_quad(600, 420, 80, 0.2)
    sc = scene(img, q, seed=3)
    halo = np.zeros((720, 1280), np.uint8)
    cv2.rectangle(halo, (520, 300), (700, 460), 255, -1)
    card = np.zeros((720, 1280), np.uint8)
    cv2.fillConvexPoly(card, np.int32(q), 255)
    sc[(halo > 0) & (card == 0)] = (225, 228, 230)                    # pale paper-like surroundings
    box = [600 - 70, 380 - 70, 600 + 70, 380 + 70]                    # off-centre, larger than the card
    res = rec.classify_region(sc, box, card_h=85)
    assert res and res[0].cls == cls


def test_blank_back_is_not_a_target(rec):
    back = np.full((400, 300, 3), 235, np.uint8)
    found = rec.find_all(scene(back, card_quad(640, 420, 150)), include_backs=True)
    assert found and all(f.back for f in found)
    assert not rec.find_all(scene(back, card_quad(640, 420, 150)))


def test_no_cards_in_empty_scene(rec):
    img = scene(np.full((400, 300, 3), 120, np.uint8), card_quad(640, 420, 10))
    assert rec.find_all(img) == []


@pytest.mark.parametrize('gain', [0.6, 1.0])
def test_star_colour_under_lighting(gain):
    for name, cls, star, img in refs():
        if cls != 'SAFE':
            continue
        lit = np.clip(img.astype(float) * gain * np.array([1.1, 1.0, 0.85]), 0, 255).astype(np.uint8)  # warm, dim
        colour, _ = star_from_card(cv2, cv2.resize(lit, (60, 80), interpolation=cv2.INTER_AREA))
        assert colour == (star or 'none'), '%s gain %.1f -> %s' % (name, gain, colour)
