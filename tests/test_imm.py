"""Tests for the Identity Mixture Model."""

import numpy as np
import pytest

from axiom.models.imm import IdentityMixtureModel


def _make_clustered_features(
    points_per_cluster: int = 20,
    include_third: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    centers = [
        np.array([1.0, 0.0, 0.0, 0.03, 0.05]),
        np.array([0.0, 1.0, 0.0, 0.17, 0.06]),
    ]
    if include_third:
        centers.append(np.array([0.0, 0.0, 1.0, 0.05, 0.19]))

    features = []
    labels = []
    for idx, center in enumerate(centers):
        noise_scale = np.array([0.03, 0.03, 0.03, 0.01, 0.01])
        cluster = center + rng.normal(0.0, noise_scale, size=(points_per_cluster, 5))
        cluster[:, :3] = np.clip(cluster[:, :3], 0.0, 1.0)
        cluster[:, 3:] = np.clip(cluster[:, 3:], 1e-4, 1.0)
        features.append(cluster)
        labels.append(np.full(points_per_cluster, idx, dtype=int))
    return np.vstack(features), np.concatenate(labels)


def _run_inference(imm: IdentityMixtureModel, features: np.ndarray, steps: int = 8) -> np.ndarray:
    assignments = np.zeros(features.shape[0], dtype=int)
    for _ in range(steps):
        assignments = imm.infer_identity(features)
    return assignments


class TestIdentityMixtureModel:
    def test_initialization(self):
        imm = IdentityMixtureModel(max_types=5, alpha_imm=1.0)
        assert imm.max_types == 5
        assert imm.num_active_types == 0

    def test_identity_assignment_for_distinct_clusters(self):
        features, labels = _make_clustered_features(points_per_cluster=24, include_third=True)
        imm = IdentityMixtureModel(max_types=6, alpha_imm=0.6)
        assignments = _run_inference(imm, features, steps=10)

        assert imm.num_active_types >= 3
        dominant_types = []
        for cluster_id in np.unique(labels):
            pred_cluster = assignments[labels == cluster_id]
            vals, counts = np.unique(pred_cluster, return_counts=True)
            dominant_idx = int(np.argmax(counts))
            dominant_type = int(vals[dominant_idx])
            dominant_types.append(dominant_type)
            purity = counts[dominant_idx] / pred_cluster.size
            assert purity > 0.8
        assert len(set(dominant_types)) == 3

    def test_niw_update_matches_hand_computation_single_point(self):
        imm = IdentityMixtureModel(max_types=3, alpha_imm=1.0)
        imm.base_m = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        imm.base_kappa = 2.0
        imm.base_n = 9.0
        imm.base_u = np.diag([0.5, 0.6, 0.7, 0.8, 0.9]).astype(np.float64)

        imm.num_active_types = 1
        imm.niw_params = [
            {
                "m": np.zeros(5),
                "kappa": 1.0,
                "u": np.eye(5),
                "n": 7.0,
                "weight": 1.0,
            }
        ]

        x = np.array([[0.7, 0.1, 0.2, 0.3, 0.4]])
        imm.update_params(x, np.array([0]))
        posterior = imm.niw_params[0]

        expected_kappa = imm.base_kappa + 1.0
        expected_n = imm.base_n + 1.0
        expected_m = (imm.base_kappa * imm.base_m + x[0]) / expected_kappa
        diff = x[0] - imm.base_m
        expected_u = imm.base_u + (imm.base_kappa / expected_kappa) * np.outer(diff, diff)

        assert posterior["kappa"] == pytest.approx(expected_kappa)
        assert posterior["n"] == pytest.approx(expected_n)
        assert np.allclose(posterior["m"], expected_m)
        assert np.allclose(posterior["u"], expected_u)

    def test_type_expansion_when_novel_type_appears(self):
        imm = IdentityMixtureModel(max_types=6, alpha_imm=0.5)

        features_old, _ = _make_clustered_features(points_per_cluster=18, include_third=False)
        _run_inference(imm, features_old, steps=1)
        types_before = imm.num_active_types

        features_new, _ = _make_clustered_features(points_per_cluster=18, include_third=True)
        _run_inference(imm, features_new, steps=3)
        types_after = imm.num_active_types

        assert types_after > types_before
