"""Classical anomaly detection baselines, fit on healthy data only, to
compare the graph autoencoder against. See TowerWatch_guideline.md
Sec 6.1: "do not write the GNN before the baselines exist."

Both operate on a flattened per-window feature vector (every sensor's
features concatenated into one row), the same information the GNN sees,
just without the sensor graph structure telling it which entries are
structurally related.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class PCABaseline:
    """Reconstruction-error novelty detection: fit PCA on healthy data,
    score new samples by how poorly PCA's reduced representation
    reconstructs them. The classic SHM baseline.
    """
    n_components: int
    pca: PCA = None

    def fit(self, x_healthy_train: np.ndarray) -> "PCABaseline":
        self.pca = PCA(n_components=self.n_components).fit(x_healthy_train)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per sample (higher = more anomalous)."""
        reconstructed = self.pca.inverse_transform(self.pca.transform(x))
        return np.mean((x - reconstructed) ** 2, axis=1)


@dataclass
class IsolationForestBaseline:
    """Isolation Forest anomaly detection, fit on healthy data only."""
    n_estimators: int = 100
    random_state: int = 0
    model: IsolationForest = None

    def fit(self, x_healthy_train: np.ndarray) -> "IsolationForestBaseline":
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state,
        ).fit(x_healthy_train)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score per sample, higher = more anomalous (sklearn's
        decision_function is the opposite convention, so it's negated here
        to match PCABaseline and the GNN's score direction).
        """
        return -self.model.decision_function(x)
