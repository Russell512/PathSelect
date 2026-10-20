"""Synthetic four-stage example. Its numbers are not paper results."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import PathSelect


def run(seed=7):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(seed)
    texts = F.normalize(torch.randn(4, 2, 512, generator=generator), dim=-1)
    train, test = {}, {}
    for p in range(4):
        def slides(n):
            labels = torch.arange(n) % 2 + 2 * p
            z = [texts[p, int(y) % 2] * 2 +
                 0.5 * torch.randn(96, 512, generator=generator) for y in labels]
            return z, labels
        train[p], test[p] = slides(8), slides(4)
    records = []
    for order, tasks in [("paper", [3, 2, 1, 0]), ("reverse", [0, 1, 2, 3])]:
        model, seen = PathSelect(texts), []
        for stage, p in enumerate(tasks, 1):
            model.fit_task(p, *train[p])
            seen.append(p)
            accuracy = []
            for q in seen:
                zs, ys = test[q]
                preds = [model.predict_slide(z) for z in zs]
                assert all(len(r["indices"]) == 64 for r in preds)
                accuracy.append(sum(r["predicted_class"] == int(y)
                                    for r, y in zip(preds, ys)) / len(ys))
            records.append({"order": order, "stage": stage,
                            "synthetic_task_macro_accuracy": sum(accuracy) / len(accuracy)})
    return {"kind": "synthetic demonstration, not paper evidence", "seed": seed,
            "stages": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = json.dumps(run(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
