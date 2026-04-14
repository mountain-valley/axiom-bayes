"""
Identity Mixture Model (iMM).

Infers a discrete identity code for each object slot based on its color and shape
features. Models {c^(k), e^(k)} as a mixture of up to V Gaussians with
Normal-Inverse-Wishart priors. Enables type-specific (not instance-specific) dynamics
learning and robustness to color/shape perturbations.

Reference: AXIOM paper Section 2, Equations 3-4.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


class IdentityMixtureModel:
    """Gaussian mixture over slot color+shape features, producing discrete identity codes."""

    def __init__(self, max_types: int, alpha_imm: float = 1.0):
        self.max_types = max_types
        self.alpha_imm = alpha_imm
        self.num_active_types = 0
        self.niw_params: list[dict] = []
        self.feature_dim = 5
        self.base_m = np.zeros(self.feature_dim, dtype=np.float64)
        self.base_kappa = 1.0
        self.base_n = self.feature_dim + 2.0
        self.base_u = np.eye(self.feature_dim, dtype=np.float64) * 0.25
        self.novelty_distance_threshold = 0.35
        self.min_novel_points = 4
        self.distance_penalty_scale = 0.2

    def infer_identity(self, color_shape_features: np.ndarray) -> np.ndarray:
        """
        Assign each slot to an identity type.

        Args:
            color_shape_features: (K, 5) array of [c, e] per slot.

        Returns:
            (K,) integer array of identity codes.
        """
        if color_shape_features.ndim != 2 or color_shape_features.shape[1] != self.feature_dim:
            raise ValueError("color_shape_features must have shape (K, 5)")
        if color_shape_features.shape[0] == 0:
            return np.zeros(0, dtype=int)

        if self.num_active_types == 0:
            self._add_type_from_data(color_shape_features[:1])

        responsibilities, _ = self._e_step(color_shape_features)
        self.update_params(color_shape_features, responsibilities)
        prev_types = self.num_active_types
        self.expand_if_needed(color_shape_features)

        if self.num_active_types != prev_types:
            responsibilities, _ = self._e_step(color_shape_features)
            self.update_params(color_shape_features, responsibilities)

        return np.argmax(responsibilities, axis=1)

    def update_params(self, color_shape_features: np.ndarray, assignments: np.ndarray):
        """M-step: update NIW parameters for each identity component."""
        if self.num_active_types == 0:
            return

        num_slots = color_shape_features.shape[0]
        num_types = self.num_active_types
        if assignments.ndim == 1:
            if assignments.shape[0] != num_slots:
                raise ValueError("Hard assignments must have shape (K,)")
            responsibilities = np.zeros((num_slots, num_types), dtype=np.float64)
            responsibilities[np.arange(num_slots), assignments] = 1.0
        elif assignments.ndim == 2:
            responsibilities = assignments.astype(np.float64, copy=False)
            if responsibilities.shape != (num_slots, num_types):
                raise ValueError("Soft assignments must have shape (K, num_active_types)")
        else:
            raise ValueError("assignments must be either (K,) or (K, J)")

        norm = responsibilities.sum(axis=1, keepdims=True) + 1e-12
        responsibilities = responsibilities / norm

        counts = responsibilities.sum(axis=0)
        priors = self._dirichlet_prior(num_types)
        weights = counts + priors
        weights /= np.sum(weights)

        for type_idx in range(num_types):
            count = float(counts[type_idx])
            if count < 1e-9:
                self.niw_params[type_idx]["weight"] = float(weights[type_idx])
                continue

            r = responsibilities[:, type_idx]
            x_bar = (r[:, None] * color_shape_features).sum(axis=0) / count
            centered = color_shape_features - x_bar[None, :]
            s_mat = (r[:, None] * centered).T @ centered

            prior_m = self.base_m
            prior_kappa = self.base_kappa
            prior_n = self.base_n
            prior_u = self.base_u

            kappa_n = prior_kappa + count
            m_n = (prior_kappa * prior_m + count * x_bar) / kappa_n
            n_n = prior_n + count

            diff = x_bar - prior_m
            u_n = prior_u + s_mat + ((prior_kappa * count) / kappa_n) * np.outer(diff, diff)

            self.niw_params[type_idx].update(
                {
                    "m": m_n,
                    "kappa": float(kappa_n),
                    "u": u_n,
                    "n": float(n_n),
                    "weight": float(weights[type_idx]),
                }
            )

    def expand_if_needed(self, color_shape_features: np.ndarray):
        """Grow a new identity type if existing ones can't explain the data."""
        if self.num_active_types >= self.max_types:
            return
        if self.num_active_types == 0:
            self._add_type_from_data(color_shape_features[:1])
            return

        component_means = np.stack(
            [params["m"] for params in self.niw_params[: self.num_active_types]],
            axis=0,
        )
        distances = np.linalg.norm(
            color_shape_features[:, None, :] - component_means[None, :, :],
            axis=2,
        )
        min_distance = np.min(distances, axis=1)
        novel_mask = min_distance > self.novelty_distance_threshold
        if not np.any(novel_mask):
            return
        if int(np.sum(novel_mask)) < self.min_novel_points:
            return

        candidate_data = color_shape_features[novel_mask]
        self._add_type_from_data(candidate_data)

    def _dirichlet_prior(self, num_types: int) -> np.ndarray:
        prior = np.ones(num_types, dtype=np.float64)
        prior[-1] = self.alpha_imm
        return prior

    def _e_step(self, color_shape_features: np.ndarray) -> tuple[np.ndarray, float]:
        # For identity clustering, use posterior-predictive fit for responsibilities
        # and update mixture weights in the M-step from component counts.
        scores = self._component_log_likelihoods(color_shape_features, include_weights=False)
        means = np.stack(
            [params["m"] for params in self.niw_params[: self.num_active_types]], axis=0
        )
        dist2 = np.sum(
            (color_shape_features[:, None, :] - means[None, :, :]) ** 2,
            axis=2,
        )
        scores = scores - dist2 / (2.0 * self.distance_penalty_scale**2)
        norm = logsumexp(scores, axis=1, keepdims=True)
        responsibilities = np.exp(scores - norm)
        return responsibilities, float(np.sum(norm))

    def _component_log_likelihoods(
        self,
        color_shape_features: np.ndarray,
        include_weights: bool,
    ) -> np.ndarray:
        scores = np.empty((color_shape_features.shape[0], self.num_active_types), dtype=np.float64)
        for type_idx, params in enumerate(self.niw_params[: self.num_active_types]):
            df, loc, scale = self._predictive_student_t_params(params)
            ll = self._log_multivariate_student_t(color_shape_features, df, loc, scale)
            if include_weights:
                ll = ll + np.log(max(float(params["weight"]), 1e-12))
            scores[:, type_idx] = ll
        return scores

    def _predictive_student_t_params(
        self, params: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        d = self.feature_dim
        kappa = float(params["kappa"])
        n = float(params["n"])
        df = max(n - d + 1.0, 1e-6)
        loc = np.asarray(params["m"], dtype=np.float64)
        u = np.asarray(params["u"], dtype=np.float64)
        scale = ((kappa + 1.0) / (kappa * df)) * u
        return df, loc, scale

    def _log_multivariate_student_t(
        self, data: np.ndarray, df: float, loc: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        d = data.shape[1]
        jitter = 1e-8 * np.eye(d, dtype=np.float64)
        scale_reg = scale + jitter
        sign, logdet = np.linalg.slogdet(scale_reg)
        if sign <= 0:
            raise ValueError("Student-t scale matrix must be positive definite")

        inv_scale = np.linalg.inv(scale_reg)
        centered = data - loc[None, :]
        quad = np.einsum("ni,ij,nj->n", centered, inv_scale, centered)
        norm_const = (
            gammaln((df + d) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * (d * np.log(df * np.pi) + logdet)
        )
        return norm_const - 0.5 * (df + d) * np.log1p(quad / df)

    def _log_prior_predictive(self, data: np.ndarray) -> np.ndarray:
        params = {
            "m": self.base_m,
            "kappa": self.base_kappa,
            "u": self.base_u,
            "n": self.base_n,
        }
        df, loc, scale = self._predictive_student_t_params(params)
        return self._log_multivariate_student_t(data, df, loc, scale)

    def _add_type_from_data(self, data: np.ndarray):
        if self.num_active_types >= self.max_types:
            return
        if data.size == 0:
            return

        count = float(data.shape[0])
        x_bar = data.mean(axis=0)
        centered = data - x_bar[None, :]
        s_mat = centered.T @ centered

        kappa_n = self.base_kappa + count
        m_n = (self.base_kappa * self.base_m + count * x_bar) / kappa_n
        n_n = self.base_n + count
        diff = x_bar - self.base_m
        u_n = self.base_u + s_mat + ((self.base_kappa * count) / kappa_n) * np.outer(diff, diff)

        self.niw_params.append(
            {
                "m": m_n.astype(np.float64),
                "kappa": float(kappa_n),
                "u": u_n.astype(np.float64),
                "n": float(n_n),
                "weight": 1.0,
            }
        )
        self.num_active_types += 1

        priors = self._dirichlet_prior(self.num_active_types)
        weights = priors / np.sum(priors)
        for idx, weight in enumerate(weights):
            self.niw_params[idx]["weight"] = float(weight)
