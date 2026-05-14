import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from rdlm.arc import (
    ArcDataset,
    ArcStructuredDataset,
    _add_grid_noise,
    augment_structured_task,
    build_examples_from_task,
    build_structured_examples_from_task,
    collate_arc_examples,
    collate_structured_arc_examples,
    parse_grid,
    serialize_grid,
    structured_example_difficulty,
)
from rdlm.arc_model import ArcOutputDiffusion
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

    def test_structured_dataset_and_collate_preserve_grid_cells(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            dataset = ArcStructuredDataset([path])
            batch = collate_structured_arc_examples([dataset[0]])

        self.assertEqual(batch["target_colors"].shape, batch["target_mask"].shape)
        self.assertEqual(batch["target_colors"][0, 0].item(), 2)
        self.assertTrue(batch["context_mask"].any())
        self.assertEqual(batch["target_shapes"], [(1, 1)])

    def test_color_permutation_is_consistent_across_task(self):
        task = {
            "train": [{"input": [[1, 2]], "output": [[3, 4]]}],
            "test": [{"input": [[1, 3]], "output": [[2, 4]]}],
        }
        variants = augment_structured_task(task, color_permutation=True)
        augmented = variants[1]

        mapping = {
            task["train"][0]["input"][0][0]: augmented["train"][0]["input"][0][0],
            task["train"][0]["input"][0][1]: augmented["train"][0]["input"][0][1],
            task["train"][0]["output"][0][0]: augmented["train"][0]["output"][0][0],
            task["train"][0]["output"][0][1]: augmented["train"][0]["output"][0][1],
        }
        self.assertEqual(augmented["test"][0]["input"][0][0], mapping[1])
        self.assertEqual(augmented["test"][0]["input"][0][1], mapping[3])
        self.assertEqual(augmented["test"][0]["output"][0][0], mapping[2])
        self.assertEqual(augmented["test"][0]["output"][0][1], mapping[4])

    def test_translation_keeps_nonzero_cells_in_bounds(self):
        task = {
            "train": [{"input": [[0, 1, 0]], "output": [[0, 2, 0]]}],
            "test": [{"input": [[0, 3, 0]], "output": [[0, 4, 0]]}],
        }
        variants = augment_structured_task(task, translation=True)
        augmented = variants[1]

        for split in ("train", "test"):
            for pair in augmented[split]:
                for grid in (pair["input"], pair["output"]):
                    self.assertEqual(len(grid), 1)
                    self.assertEqual(len(grid[0]), 3)
                    self.assertEqual(sum(cell != 0 for row in grid for cell in row), 1)

    def test_grid_noise_only_adds_input_distractors(self):
        task = {
            "train": [{"input": [[0, 0], [0, 0]], "output": [[1, 0], [0, 0]]}],
            "test": [{"input": [[0, 0], [0, 0]], "output": [[2, 0], [0, 0]]}],
        }
        noisy_input = _add_grid_noise(task["train"][0]["input"], noise_prob=1.0)
        self.assertTrue(all(cell != 0 for row in noisy_input for cell in row))

        variants = augment_structured_task(task, grid_noise=True)
        augmented = variants[1]
        self.assertEqual(augmented["train"][0]["output"], task["train"][0]["output"])
        self.assertEqual(augmented["test"][0]["output"], task["test"][0]["output"])

    def test_structured_curriculum_sorts_by_difficulty(self):
        easy = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        hard = {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
            "test": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            hard_path = Path(tmp) / "hard.json"
            easy_path = Path(tmp) / "easy.json"
            hard_path.write_text(json.dumps(hard), encoding="utf-8")
            easy_path.write_text(json.dumps(easy), encoding="utf-8")
            dataset = ArcStructuredDataset([hard_path, easy_path], curriculum=True)

        difficulties = [structured_example_difficulty(example) for example in dataset.examples]
        self.assertEqual(difficulties, sorted(difficulties))


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

    def test_one_structured_encoder_step(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        out = model(
            context_colors=batch["context_colors"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_roles=batch["context_roles"],
            context_examples=batch["context_examples"],
            context_mask=batch["context_mask"],
            target_colors=batch["target_colors"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
        )
        out["loss"].backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(out["loss"]))

    def test_structured_refinement_checkpointing_keeps_initial_state_gradients(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(
            dim=32,
            max_grid_size=30,
            max_examples=8,
            num_heads=4,
            num_refinement_blocks=2,
            gradient_checkpointing=True,
        )

        out = model(
            context_colors=batch["context_colors"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_roles=batch["context_roles"],
            context_examples=batch["context_examples"],
            context_mask=batch["context_mask"],
            target_colors=batch["target_colors"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
        )
        out["loss"].backward()

        self.assertIsNotNone(model.output_init.grad)
        self.assertIsNotNone(model.latent_init.grad)
        self.assertGreater(float(model.output_init.grad.norm()), 0.0)
        self.assertGreater(float(model.latent_init.grad.norm()), 0.0)

    def test_structured_aux_loss_backward(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(
            dim=32,
            max_grid_size=30,
            max_examples=8,
            num_heads=4,
            num_refinement_blocks=3,
            aux_loss_weight=0.25,
        )

        out = model(
            context_colors=batch["context_colors"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_roles=batch["context_roles"],
            context_examples=batch["context_examples"],
            context_mask=batch["context_mask"],
            target_colors=batch["target_colors"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
        )
        out["loss"].backward()

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["aux_loss"]))

    def test_structured_ensemble_single_candidate_matches_greedy(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)

        torch.manual_seed(11)
        greedy = model.sample(
            context_colors=batch["context_colors"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_roles=batch["context_roles"],
            context_examples=batch["context_examples"],
            context_mask=batch["context_mask"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
            steps=4,
        )
        torch.manual_seed(11)
        ensemble = model.sample_ensemble(
            context_colors=batch["context_colors"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_roles=batch["context_roles"],
            context_examples=batch["context_examples"],
            context_mask=batch["context_mask"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
            steps=4,
            num_candidates=1,
            temperature_start=0.0,
            temperature_end=0.0,
        )

        self.assertTrue(torch.equal(ensemble, greedy))

    def test_structured_ensemble_preserves_target_mask(self):
        model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)
        zeros = torch.zeros((1, 1), dtype=torch.long)
        target_rows = torch.tensor([[0, 0]], dtype=torch.long)
        target_cols = torch.tensor([[0, 1]], dtype=torch.long)
        target_mask = torch.tensor([[True, False]])

        def fake_sample_with_scores(self, **kwargs):
            return (
                torch.tensor([[7, 8]], dtype=torch.long),
                torch.tensor([[-0.2, -0.1]], dtype=torch.float),
            )

        model._sample_with_scores = types.MethodType(fake_sample_with_scores, model)
        generated = model.sample_ensemble(
            context_colors=zeros,
            context_rows=zeros,
            context_cols=zeros,
            context_roles=zeros,
            context_examples=zeros,
            context_mask=torch.ones((1, 1), dtype=torch.bool),
            target_rows=target_rows,
            target_cols=target_cols,
            target_mask=target_mask,
            num_candidates=1,
        )

        self.assertEqual(generated.tolist(), [[7, 0]])

    def test_structured_majority_tie_uses_log_prob_then_lowest_color(self):
        model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)
        zeros = torch.zeros((1, 1), dtype=torch.long)
        target_mask = torch.ones((1, 1), dtype=torch.bool)
        candidates = iter([
            (torch.tensor([[3]]), torch.tensor([[-0.4]])),
            (torch.tensor([[4]]), torch.tensor([[-0.2]])),
            (torch.tensor([[3]]), torch.tensor([[-0.6]])),
            (torch.tensor([[4]]), torch.tensor([[-0.9]])),
        ])

        def fake_sample_with_scores(self, **kwargs):
            return next(candidates)

        model._sample_with_scores = types.MethodType(fake_sample_with_scores, model)
        generated = model.sample_ensemble(
            context_colors=zeros,
            context_rows=zeros,
            context_cols=zeros,
            context_roles=zeros,
            context_examples=zeros,
            context_mask=torch.ones((1, 1), dtype=torch.bool),
            target_rows=zeros,
            target_cols=zeros,
            target_mask=target_mask,
            num_candidates=4,
            strategy="majority",
        )
        self.assertEqual(generated.item(), 3)

        candidates = iter([
            (torch.tensor([[3]]), torch.tensor([[-0.5]])),
            (torch.tensor([[4]]), torch.tensor([[-0.5]])),
        ])
        generated = model.sample_ensemble(
            context_colors=zeros,
            context_rows=zeros,
            context_cols=zeros,
            context_roles=zeros,
            context_examples=zeros,
            context_mask=torch.ones((1, 1), dtype=torch.bool),
            target_rows=zeros,
            target_cols=zeros,
            target_mask=target_mask,
            num_candidates=2,
            strategy="majority",
        )
        self.assertEqual(generated.item(), 3)


if __name__ == "__main__":
    unittest.main()
