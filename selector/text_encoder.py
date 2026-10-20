"""Read the original frozen text caches; no foundation-model dependency."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _REPO_ROOT / "configs/base.yaml"


def _abs(path):
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def load_config(path=None, machine=None):
    """Read portable defaults, optional ignored site settings, then data-root env."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    machine = machine or os.environ.get("NAVCIL_MACHINE")
    if machine:
        cfg.update(yaml.safe_load(
            (_REPO_ROOT / "configs" / f"machine_{machine}.yaml").read_text()) or {})
        cfg["machine"] = machine
    if os.environ.get("PATHSELECT_DATA_ROOT"):
        cfg["dataset_root_dir"] = str(Path(os.environ["PATHSELECT_DATA_ROOT"]).resolve())
    return cfg


@dataclass
class ClassTextFeatures:
    f_txt: torch.Tensor
    logit_scale: torch.Tensor
    class_names: list[str]
    task: str

    def to(self, device):
        return ClassTextFeatures(self.f_txt.to(device), self.logit_scale.to(device),
                                 self.class_names, self.task)


def class_prompt_ensemble(task, class_prompt_path):
    spec = json.loads(_abs(class_prompt_path).read_text())["0"]
    if task not in spec["classnames"]:
        raise KeyError(task)
    names, groups = [], []
    for subtype, synonyms in spec["classnames"][task].items():
        names.append(subtype)
        groups.append([t.replace("CLASSNAME", name) + "."
                       for name in synonyms for t in spec["templates"]])
    return names, groups


def build_f_txt(task, cfg=None, *, device="cpu", refresh=False):
    if refresh:
        raise ValueError("This release uses frozen caches; do not regenerate paper texts.")
    cfg = cfg or load_config()
    blob = torch.load(_abs(cfg["f_txt_cache_dir"]) / f"f_txt_{task}.pt",
                      map_location="cpu", weights_only=True)
    return ClassTextFeatures(blob["f_txt"].to(device), blob["logit_scale"].to(device),
                             list(blob["class_names"]), task)
