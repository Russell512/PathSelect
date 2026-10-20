"""Small CPU inference/training API around the validated experiment primitives."""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.paper64_common import Readout, evidence, feature_check, mean_norm, route
from scripts.paper64_run import predict
from selector.incremental_ridge import IncrementalRidge


class PathSelect:
    """FINAL-B with incremental task/subtype ridge and one-shot top-K selection.

    ``texts[p]`` has shape [2, 512]; task p owns global classes 2p and 2p+1.
    Train labels are global class IDs. Inference receives only patch features.
    Full A-head training, paired controls and validation are in the study runner.
    """

    def __init__(self, texts, gamma=1e-3, gamma_router=1e-3, budget=64):
        self.texts = torch.as_tensor(texts).detach().cpu().float().clone()
        if self.texts.ndim != 3 or self.texts.shape[1:] != (2, 512):
            raise ValueError("texts must have shape [number_of_tasks, 2, 512]")
        if not torch.isfinite(self.texts).all():
            raise ValueError("nonfinite texts")
        if not torch.allclose(self.texts.norm(dim=-1), torch.ones(self.texts.shape[:2]),
                              atol=1e-5, rtol=1e-5):
            raise ValueError("supply the original L2-normalized class texts")
        if gamma <= 0 or gamma_router <= 0 or not isinstance(budget, int) or budget < 1:
            raise ValueError("ridge gammas and integer budget must be positive")
        self.budget = budget
        self.router = IncrementalRidge(gamma_router)
        self.readout = Readout(gamma)
        self._router_state = self._subtype_state = None

    def _features(self, patches):
        z = torch.as_tensor(patches).detach().cpu().float()
        feature_check(z, "input slide")
        return z

    def _evidence(self, z, task):
        # Selecting all patches avoids an undefined sample SD for a singleton.
        if len(z) <= self.budget:
            idx = torch.arange(len(z))
            return mean_norm(z), idx
        from selector.flat_selector import text_nav_feats
        from selector.i6_expert import zscore
        from scripts.paper64_common import select_indices
        score = zscore(text_nav_feats(z, self.texts[task])[:, 0])
        idx = select_indices(z, score, "top64", self.budget)
        return mean_norm(z, idx), idx

    @torch.no_grad()
    def fit_task(self, task, slides, labels):
        """Add one new task using all provided training slides; no replay buffer."""
        if task not in range(len(self.texts)) or task in self.router.task_ids:
            raise ValueError("task must be a valid, previously unseen task ID")
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.ndim != 1 or set(labels.tolist()) != {2 * task, 2 * task + 1}:
            raise ValueError("training labels must contain both global task classes")
        means, selected = [], []
        for patches in slides:
            z = self._features(patches)
            means.append(mean_norm(z))
            selected.append(self._evidence(z, task)[0])
        if len(means) != len(labels):
            raise ValueError("one label is required per training WSI")
        self.router.add_task(task, torch.stack(means))
        self.readout.add_task(task, torch.stack(selected), labels)
        self._router_state = {"W": self.router.solve(), "tasks": list(self.router.task_ids)}
        self._subtype_state = self.readout.state()
        return self

    @torch.no_grad()
    def predict_slide(self, patches, readout="ridge"):
        """Route, select once, then classify within the predicted task's pair."""
        if self._router_state is None:
            raise ValueError("fit at least one task before prediction")
        if readout not in {"ridge", "cosine"}:
            raise ValueError("readout must be ridge or cosine")
        z = self._features(patches)
        task, router_scores = route(z, self._router_state)
        ev, indices = self._evidence(z, task)
        pred, scores = predict(ev[None], torch.tensor([task]), self._subtype_state,
                               self.texts.flatten(0, 1),
                               "RDG" if readout == "ridge" else "COS")
        return {"predicted_task": task, "predicted_class": int(pred[0]),
                "indices": indices, "evidence": ev, "scores": scores[0],
                "class_columns": self._subtype_state["classes"],
                "router_scores": router_scores}

    def save(self, path):
        """Save sufficient statistics and frozen inference state; no training WSIs."""
        if self._router_state is None:
            raise ValueError("cannot save an unfitted model")
        torch.save({"format": "pathselect-b-v1", "texts": self.texts,
                    "budget": self.budget, "gamma": self.readout.gamma,
                    "gamma_router": self.router.gamma, "router_A": self.router.A,
                    "router_b": self.router.b, "router_tasks": self.router.task_ids,
                    "subtype": self._subtype_state}, Path(path))

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d["format"] != "pathselect-b-v1":
            raise ValueError("unknown model format")
        model = cls(d["texts"], d["gamma"], d["gamma_router"], d["budget"])
        model.router.A, model.router.b = d["router_A"], d["router_b"]
        model.router.task_ids = d["router_tasks"]
        state = d["subtype"]
        model.readout.A, model.readout.tasks = state["A"], state["tasks"]
        model.readout.cols = {c: state["B"][:, i] for i, c in enumerate(state["classes"])}
        model._subtype_state = state
        model._router_state = {"W": model.router.solve(), "tasks": list(model.router.task_ids)}
        return model
