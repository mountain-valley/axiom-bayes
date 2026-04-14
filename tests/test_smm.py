"""Tests for the Slot Mixture Model."""

import numpy as np

from axiom.models.smm import SlotMixtureModel
from envs.utils import image_to_tokens


def _make_rect_scene(
    height: int = 24,
    width: int = 24,
    include_green: bool = True,
    include_blue: bool = True,
) -> tuple[np.ndarray, dict]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    regions = {
        "red": ((2, 10), (2, 9), np.array([255, 0, 0], dtype=np.uint8)),
    }
    if include_green:
        regions["green"] = ((3, 11), (14, 21), np.array([0, 255, 0], dtype=np.uint8))
    if include_blue:
        regions["blue"] = ((14, 21), (8, 16), np.array([0, 0, 255], dtype=np.uint8))

    for (r0, r1), (c0, c1), color in regions.values():
        image[r0:r1, c0:c1] = color
    return image, regions


def _region_mask(shape: tuple[int, int], region: tuple[tuple[int, int], tuple[int, int], np.ndarray]):
    (r0, r1), (c0, c1), _ = region
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask.ravel()


def _run_em(smm: SlotMixtureModel, tokens: np.ndarray, n_steps: int = 8) -> dict:
    result = {}
    for _ in range(n_steps):
        result = smm.infer(tokens)
    return result


class TestSlotMixtureModel:
    def test_initialization(self):
        smm = SlotMixtureModel(max_slots=8, alpha_smm=1.0)
        smm.initialize((24, 24, 3))
        assert smm.max_slots == 8
        assert smm.num_active_slots == 0
        assert smm.a_matrix.shape == (5, 7)
        assert smm.b_matrix.shape == (2, 7)

    def test_assignments_on_synthetic_rectangles(self):
        image, regions = _make_rect_scene(include_green=True, include_blue=True)
        tokens = image_to_tokens(image)

        smm = SlotMixtureModel(max_slots=8, alpha_smm=0.6)
        smm.initialize(image.shape)
        result = _run_em(smm, tokens, n_steps=10)

        responsibilities = result["responsibilities"]
        assert smm.num_active_slots >= 4  # 3 objects + background

        expected_colors = {
            name: color.astype(np.float64) / 255.0 for name, (_, _, color) in regions.items()
        }
        slot_colors = np.stack(
            [slot["color"] for slot in smm.slot_params[: smm.num_active_slots]],
            axis=0,
        )

        for name, region in regions.items():
            mask = _region_mask(image.shape[:2], region)
            region_resp = responsibilities[mask].mean(axis=0)
            dominant_slot = int(np.argmax(region_resp))
            assert region_resp[dominant_slot] > 0.6
            assert np.linalg.norm(slot_colors[dominant_slot] - expected_colors[name]) < 0.35

    def test_parameter_recovery_after_em_iterations(self):
        image, regions = _make_rect_scene(include_green=False, include_blue=False)
        tokens = image_to_tokens(image)

        smm = SlotMixtureModel(max_slots=4, alpha_smm=0.7)
        smm.initialize(image.shape)
        _run_em(smm, tokens, n_steps=10)

        red_rgb = np.array([1.0, 0.0, 0.0])
        slot_colors = np.stack(
            [slot["color"] for slot in smm.slot_params[: smm.num_active_slots]],
            axis=0,
        )
        red_slot = int(np.argmin(np.linalg.norm(slot_colors - red_rgb[None, :], axis=1)))
        red_slot_params = smm.slot_params[red_slot]

        red_region = regions["red"]
        red_mask = _region_mask(image.shape[:2], red_region)
        red_tokens = tokens[red_mask]
        expected_pos = red_tokens[:, :2].mean(axis=0)
        expected_color = red_tokens[:, 2:].mean(axis=0)
        expected_extent = np.var(red_tokens[:, :2], axis=0)

        assert np.allclose(red_slot_params["position"], expected_pos, atol=0.2)
        assert np.allclose(red_slot_params["color"], expected_color, atol=0.2)
        assert np.allclose(red_slot_params["extent"], expected_extent, atol=0.15)

    def test_slot_expansion_when_new_object_appears(self):
        image_a, _ = _make_rect_scene(include_green=False, include_blue=False)
        image_b, _ = _make_rect_scene(include_green=True, include_blue=False)

        smm = SlotMixtureModel(max_slots=6, alpha_smm=0.5)
        smm.initialize(image_a.shape)

        tokens_a = image_to_tokens(image_a)
        _run_em(smm, tokens_a, n_steps=6)
        slots_before = smm.num_active_slots

        tokens_b = image_to_tokens(image_b)
        _run_em(smm, tokens_b, n_steps=6)
        slots_after = smm.num_active_slots

        assert slots_after > slots_before
