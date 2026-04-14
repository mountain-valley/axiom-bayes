"""
Slot Mixture Model (sMM).

Parses RGB image frames into object-centric latent representations. Each image is
tokenized into HxW pixel tokens, and each token is assigned to one of K Gaussian
mixture components (slots). Slot parameters encode object position, color, and
spatial extent. Uses a truncated stick-breaking prior for automatic slot expansion.

Reference: AXIOM paper Section 2, Equation 2.
"""

import numpy as np
from scipy.special import logsumexp


class SlotMixtureModel:
    """Gaussian mixture over pixel tokens, parameterized by object-centric slot latents."""

    def __init__(self, max_slots: int, alpha_smm: float = 1.0):
        self.max_slots = max_slots
        self.alpha_smm = alpha_smm
        self.num_active_slots = 0
        self.slot_params: list[dict] = []
        self.a_matrix: np.ndarray | None = None
        self.b_matrix: np.ndarray | None = None
        self.image_shape: tuple[int, int, int] | None = None
        self.min_spatial_var = 1e-4
        self.min_color_var = 1e-4
        self.expansion_min_pixels = 8

    def initialize(self, image_shape: tuple[int, int, int]):
        """Set up fixed projection matrices A, B given image dimensions."""
        if len(image_shape) != 3 or image_shape[2] != 3:
            raise ValueError("image_shape must be (H, W, 3)")

        h, w, _ = image_shape
        if h < 2 or w < 2:
            raise ValueError("image dimensions must both be at least 2")

        self.image_shape = image_shape

        # x^(k) = [p_x, p_y, c_r, c_g, c_b, e_x, e_y]
        self.a_matrix = np.zeros((5, 7), dtype=np.float64)
        self.a_matrix[0, 0] = 1.0
        self.a_matrix[1, 1] = 1.0
        self.a_matrix[2, 2] = 1.0
        self.a_matrix[3, 3] = 1.0
        self.a_matrix[4, 4] = 1.0

        self.b_matrix = np.zeros((2, 7), dtype=np.float64)
        self.b_matrix[0, 5] = 1.0
        self.b_matrix[1, 6] = 1.0

        dx = 2.0 / (w - 1)
        dy = 2.0 / (h - 1)
        self.min_spatial_var = 0.25 * min(dx, dy) ** 2
        self.min_color_var = 1e-4

    def infer(self, pixel_tokens: np.ndarray) -> dict:
        """
        E-step: assign pixel tokens to slots and update slot latents.

        Returns dict with slot assignments and updated slot features.
        """
        if pixel_tokens.ndim != 2 or pixel_tokens.shape[1] != 5:
            raise ValueError("pixel_tokens must have shape (N, 5)")
        if self.a_matrix is None or self.b_matrix is None:
            raise RuntimeError("Call initialize(image_shape) before infer().")

        if self.num_active_slots == 0:
            self._add_slot_from_pixels(pixel_tokens)

        responsibilities, log_likelihood = self._e_step(pixel_tokens)
        self.update_params(pixel_tokens, responsibilities)
        prev_slots = self.num_active_slots
        self.expand_if_needed(pixel_tokens)

        # If structure expanded, run another E/M pass so returned assignments
        # are consistent with the newly added slot.
        if self.num_active_slots != prev_slots:
            responsibilities, log_likelihood = self._e_step(pixel_tokens)
            self.update_params(pixel_tokens, responsibilities)

        assignments = np.argmax(responsibilities, axis=1)

        return {
            "assignments": assignments,
            "responsibilities": responsibilities,
            "log_likelihood": float(log_likelihood),
            "slot_features": self._slot_feature_matrix(),
            "num_active_slots": self.num_active_slots,
        }

    def update_params(self, pixel_tokens: np.ndarray, assignments: np.ndarray):
        """M-step: update slot parameters given current assignments."""
        if self.num_active_slots == 0:
            return

        n_pixels = pixel_tokens.shape[0]
        k_active = self.num_active_slots

        if assignments.ndim == 1:
            if assignments.shape[0] != n_pixels:
                raise ValueError("Hard assignments must have shape (N,)")
            responsibilities = np.zeros((n_pixels, k_active), dtype=np.float64)
            responsibilities[np.arange(n_pixels), assignments] = 1.0
        elif assignments.ndim == 2:
            responsibilities = assignments.astype(np.float64, copy=False)
            if responsibilities.shape != (n_pixels, k_active):
                raise ValueError(
                    "Soft assignments must have shape (N, num_active_slots)"
                )
        else:
            raise ValueError("assignments must be either (N,) or (N, K)")

        normalizer = responsibilities.sum(axis=1, keepdims=True) + 1e-12
        responsibilities = responsibilities / normalizer

        counts = responsibilities.sum(axis=0)
        priors = self._dirichlet_prior(k_active)
        weights = counts + priors
        weights = weights / np.sum(weights)

        coords = pixel_tokens[:, :2]
        colors = pixel_tokens[:, 2:]

        for slot_idx in range(k_active):
            slot_weight = responsibilities[:, slot_idx]
            slot_count = counts[slot_idx]
            if slot_count < 1e-8:
                self.slot_params[slot_idx]["weight"] = float(weights[slot_idx])
                continue

            inv_count = 1.0 / slot_count
            position = np.sum(slot_weight[:, None] * coords, axis=0) * inv_count
            color = np.sum(slot_weight[:, None] * colors, axis=0) * inv_count

            centered_coords = coords - position[None, :]
            coord_var = (
                np.sum(slot_weight[:, None] * centered_coords**2, axis=0) * inv_count
            )
            extent = np.maximum(coord_var, self.min_spatial_var)

            centered_colors = colors - color[None, :]
            color_var_channels = (
                np.sum(slot_weight[:, None] * centered_colors**2, axis=0) * inv_count
            )
            sigma_c = max(float(np.mean(color_var_channels)), self.min_color_var)

            self.slot_params[slot_idx].update(
                {
                    "position": position,
                    "color": np.clip(color, 0.0, 1.0),
                    "extent": extent,
                    "sigma_c": sigma_c,
                    "weight": float(weights[slot_idx]),
                }
            )

    def expand_if_needed(self, pixel_tokens: np.ndarray):
        """Check if a new slot should be created for unexplained pixels."""
        if self.num_active_slots >= self.max_slots:
            return
        if self.num_active_slots == 0:
            self._add_slot_from_pixels(pixel_tokens)
            return

        component_scores = self._component_log_likelihoods(
            pixel_tokens, include_weights=False
        )
        best_existing = np.max(component_scores, axis=1)

        prior_mean = np.mean(pixel_tokens, axis=0)
        prior_var = np.array([0.25, 0.25, 0.08, 0.08, 0.08], dtype=np.float64)
        prior_score = self._log_gaussian_diag(pixel_tokens, prior_mean, prior_var)
        threshold = prior_score + np.log(max(self.alpha_smm, 1e-12))

        slot_colors = np.stack(
            [slot["color"] for slot in self.slot_params[: self.num_active_slots]],
            axis=0,
        )
        color_diff = pixel_tokens[:, None, 2:] - slot_colors[None, :, :]
        min_color_distance = np.min(np.linalg.norm(color_diff, axis=2), axis=1)

        unexplained = (best_existing < threshold) | (min_color_distance > 0.35)

        min_pixels = max(
            self.expansion_min_pixels,
            int(np.ceil(0.01 * pixel_tokens.shape[0])),
        )
        if int(np.sum(unexplained)) < min_pixels:
            return

        self._add_slot_from_pixels(pixel_tokens[unexplained])

    def _dirichlet_prior(self, num_slots: int) -> np.ndarray:
        prior = np.ones(num_slots, dtype=np.float64)
        prior[-1] = self.alpha_smm
        return prior

    def _slot_to_mean_and_var(self, slot: dict) -> tuple[np.ndarray, np.ndarray]:
        latent = np.concatenate([slot["position"], slot["color"], slot["extent"]])
        mean = self.a_matrix @ latent
        spatial_var = np.maximum(self.b_matrix @ latent, self.min_spatial_var)
        color_var = np.full(3, max(float(slot["sigma_c"]), self.min_color_var))
        variance = np.concatenate([spatial_var, color_var], axis=0)
        return mean, variance

    def _log_gaussian_diag(self, data: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        centered = data - mean[None, :]
        log_det = np.sum(np.log(var))
        mahal = np.sum((centered**2) / var[None, :], axis=1)
        return -0.5 * (data.shape[1] * np.log(2.0 * np.pi) + log_det + mahal)

    def _e_step(self, pixel_tokens: np.ndarray) -> tuple[np.ndarray, float]:
        log_scores = self._component_log_likelihoods(pixel_tokens, include_weights=True)
        log_norm = logsumexp(log_scores, axis=1, keepdims=True)
        responsibilities = np.exp(log_scores - log_norm)
        total_log_likelihood = float(np.sum(log_norm))
        return responsibilities, total_log_likelihood

    def _component_log_likelihoods(
        self,
        pixel_tokens: np.ndarray,
        include_weights: bool,
    ) -> np.ndarray:
        k_active = self.num_active_slots
        log_scores = np.empty((pixel_tokens.shape[0], k_active), dtype=np.float64)

        for slot_idx, slot in enumerate(self.slot_params[:k_active]):
            mean, var = self._slot_to_mean_and_var(slot)
            base_score = self._log_gaussian_diag(pixel_tokens, mean, var)
            if include_weights:
                base_score = base_score + np.log(max(float(slot["weight"]), 1e-12))
            log_scores[:, slot_idx] = base_score
        return log_scores

    def _slot_feature_matrix(self) -> np.ndarray:
        if self.num_active_slots == 0:
            return np.zeros((0, 7), dtype=np.float64)
        return np.stack(
            [
                np.concatenate([slot["position"], slot["color"], slot["extent"]])
                for slot in self.slot_params[: self.num_active_slots]
            ],
            axis=0,
        )

    def _add_slot_from_pixels(self, pixel_tokens: np.ndarray):
        if self.num_active_slots >= self.max_slots:
            return
        if pixel_tokens.size == 0:
            return

        position = np.mean(pixel_tokens[:, :2], axis=0)
        color = np.mean(pixel_tokens[:, 2:], axis=0)
        extent = np.var(pixel_tokens[:, :2], axis=0)
        extent = np.maximum(extent, self.min_spatial_var)
        sigma_c = float(max(np.mean(np.var(pixel_tokens[:, 2:], axis=0)), self.min_color_var))

        self.slot_params.append(
            {
                "position": position.astype(np.float64),
                "color": np.clip(color, 0.0, 1.0).astype(np.float64),
                "extent": extent.astype(np.float64),
                "sigma_c": sigma_c,
                "weight": 1.0,
            }
        )
        self.num_active_slots += 1

        # Keep normalized mixture weights after adding a slot.
        priors = self._dirichlet_prior(self.num_active_slots)
        weights = priors / np.sum(priors)
        for slot_idx, weight in enumerate(weights):
            self.slot_params[slot_idx]["weight"] = float(weight)
