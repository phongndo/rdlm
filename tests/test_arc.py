import json
import tempfile
import unittest
from pathlib import Path

import torch

from rdlm.arc import (
    ArcDataset,
    build_examples_from_task,
    collate_arc_examples,
    parse_grid,
    serialize_grid,
)
from rdlm.diffusion_lm import RecursiveDiffusionLM
from rdlm.train_arc import create_model


class ArcSerializationTests(unittest.TestCase):
    def test_grid_round_trip(self):
        grid = [[1, 2], [3, 4]]
        text = serialize_grid(grid)
        self.assertEqual(text, "12/34")
        self.assertEqual(parse_grid(text + "~"), grid)

    def test_build_examples_skips_overlong(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        self.assertEqual(build_examples_from_task("tiny", task, seq_len=8), [])
        self.assertGreaterEqual(len(build_examples_from_task("tiny", task, seq_len=128)), 2)

    def test_dataset_and_collate_masks_answer_suffix(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            dataset = ArcDataset([path], seq_len=128)
            batch = collate_arc_examples([dataset[0]])

        self.assertEqual(batch["tokens"].shape, batch["loss_mask"].shape)
        self.assertTrue(batch["loss_mask"].any())
        first_loss_idx = int(torch.nonzero(batch["loss_mask"][0], as_tuple=False)[0].item())
        self.assertFalse(batch["loss_mask"][0, :first_loss_idx].any())


class ArcTrainingSmokeTests(unittest.TestCase):
    def test_one_conditional_diffusion_step(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        examples = build_examples_from_task("tiny", task, seq_len=128)
        batch = collate_arc_examples([examples[0]])
        model: RecursiveDiffusionLM = create_model(dim=32, seq_len=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        out = model(
            batch["tokens"],
            padding_mask=batch["padding_mask"],
            loss_mask=batch["loss_mask"],
        )
        out["loss"].backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
