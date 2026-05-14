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
    compute_grid_object_features,
    parse_grid,
    serialize_grid,
    structured_example_difficulty,
)
from rdlm.arc_model import ArcOutputDiffusion, ShapeCandidate, StructuredCandidate
from rdlm.diffusion_lm import RecursiveDiffusionLM
from rdlm.train_arc import (
    create_model,
    evaluate_structured,
    propose_mixed_shape_candidates,
    score_structured_candidate,
    structured_forward_kwargs,
    structured_prediction_metrics,
    structured_prediction_metrics_for_shapes,
)


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
        self.assertEqual(batch["target_heights"].tolist(), [1])
        self.assertEqual(batch["target_widths"].tolist(), [1])
        self.assertIn("context_object_ids", batch)
        self.assertEqual(batch["context_object_ids"].shape, batch["context_colors"].shape)

    def test_grid_object_features_are_deterministic_components(self):
        features = compute_grid_object_features([[1, 1, 0], [0, 1, 2]])

        self.assertEqual(features.object_ids, [[1, 1, 0], [0, 1, 2]])
        self.assertEqual(features.size_buckets, [[2, 2, 0], [0, 2, 1]])
        self.assertEqual(features.heights, [[2, 2, 0], [0, 2, 1]])
        self.assertEqual(features.widths, [[2, 2, 0], [0, 2, 1]])
        self.assertEqual(features.rel_rows, [[1, 1, 0], [0, 2, 1]])
        self.assertEqual(features.rel_cols, [[1, 2, 0], [0, 2, 1]])

        separated = compute_grid_object_features([[1, 0, 1]])
        self.assertEqual(separated.object_ids, [[1, 0, 2]])

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
        batch = collate_structured_arc_examples([examples[-1]])
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

    def test_one_structured_encoder_step_with_object_features(self):
        task = {
            "train": [{"input": [[1, 1], [0, 2]], "output": [[2, 2], [0, 3]]}],
            "test": [{"input": [[3, 3], [0, 4]], "output": [[4, 4], [0, 5]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(
            dim=32,
            max_grid_size=30,
            max_examples=8,
            num_heads=4,
            use_object_features=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        out = model(**structured_forward_kwargs(batch))
        out["loss"].backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(out["loss"]))

    def test_one_structured_encoder_step_with_shape_head(self):
        task = {
            "train": [{"input": [[1]], "output": [[2, 2], [0, 2]]}],
            "test": [{"input": [[3]], "output": [[4, 4], [0, 4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[0]])
        model = ArcOutputDiffusion(
            dim=32,
            max_grid_size=30,
            max_examples=8,
            num_heads=4,
            use_shape_head=True,
            shape_loss_weight=0.5,
        )

        out = model(**structured_forward_kwargs(batch))
        out["loss"].backward()

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["shape_loss"]))
        self.assertEqual(out["shape_height_logits"].shape, (1, 30))
        self.assertEqual(out["shape_width_logits"].shape, (1, 30))
        self.assertIsNotNone(model.shape_height_head.weight.grad)
        self.assertIsNotNone(model.shape_width_head.weight.grad)

    def test_structured_prediction_metrics_capture_partial_errors(self):
        generated = torch.tensor([[1, 2, 0, 0]])
        expected = torch.tensor([[1, 3, 0, 4]])
        target_mask = torch.ones((1, 4), dtype=torch.bool)
        token_log_probs = torch.log(torch.tensor([[0.9, 0.8, 0.7, 0.6]]))

        metrics = structured_prediction_metrics(
            generated,
            expected,
            target_mask,
            token_log_probs,
        )

        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["cell_count"], 4)
        self.assertEqual(metrics["correct_count"], 2)
        self.assertEqual(metrics["cell_acc"], 0.5)
        self.assertLess(metrics["nonzero_iou"], 1.0)
        self.assertGreater(metrics["color_hist_l1"], 0.0)
        self.assertAlmostEqual(metrics["mean_confidence"], 0.75, places=6)

    def test_structured_prediction_metrics_penalize_wrong_shape(self):
        metrics = structured_prediction_metrics_for_shapes(
            generated=torch.tensor([1]),
            expected=torch.tensor([1, 2, 3, 4]),
            generated_shape=(1, 1),
            expected_shape=(2, 2),
            token_log_probs=torch.log(torch.tensor([0.9])),
        )

        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["cell_count"], 4)
        self.assertEqual(metrics["correct_count"], 1)
        self.assertEqual(metrics["cell_acc"], 0.25)
        self.assertGreater(metrics["color_hist_l1"], 0.0)

    def test_mixed_shape_candidates_merge_priors_and_shape_head(self):
        task = {
            "train": [{"input": [[1]], "output": [[2, 2], [2, 2]]}],
            "test": [{"input": [[3]], "output": [[4, 4, 4], [4, 4, 4], [4, 4, 4]]}],
        }
        examples = build_structured_examples_from_task("tiny", task)
        batch = collate_structured_arc_examples([examples[-1]])
        model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)

        def fake_predict_shapes(self, **kwargs):
            return [
                [
                    ShapeCandidate(2, 2, -0.2, "shape_head"),
                    ShapeCandidate(3, 3, -0.3, "shape_head"),
                ]
            ]

        model.predict_shapes = types.MethodType(fake_predict_shapes, model)

        candidates = propose_mixed_shape_candidates(model, batch, top_k=3)

        self.assertEqual(
            [(candidate.height, candidate.width) for candidate in candidates],
            [(1, 1), (2, 2), (3, 3)],
        )
        self.assertIn("demo_output", candidates[1].source)
        self.assertIn("shape_head", candidates[1].source)

    def test_candidate_score_uses_shape_weight_and_prior_bonus(self):
        candidate = StructuredCandidate(
            height=1,
            width=1,
            sample=torch.tensor([0]),
            token_log_probs=torch.tensor([-0.5]),
            mean_token_log_prob=-1.0,
            shape_log_prob=-2.0,
            source="query_input+shape_head",
        )

        self.assertAlmostEqual(score_structured_candidate(candidate, 0.25), -1.45)

    def test_inferred_shape_eval_does_not_use_target_shape_for_candidates(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4, 4], [4, 4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            dataset = ArcStructuredDataset([path])
            dataset.examples = [dataset.examples[-1]]
            model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)
            recorded_shapes: list[tuple[int, int]] = []

            def fake_sample_with_scores(self, target_rows, **kwargs):
                return (
                    torch.zeros_like(target_rows),
                    torch.zeros_like(target_rows, dtype=torch.float),
                )

            def fake_sample_candidates(self, **kwargs):
                candidate_shapes = kwargs["candidate_shapes"]
                recorded_shapes.extend(
                    (candidate.height, candidate.width) for candidate in candidate_shapes
                )
                candidate = candidate_shapes[0]
                return [
                    StructuredCandidate(
                        height=candidate.height,
                        width=candidate.width,
                        sample=torch.tensor([0]),
                        token_log_probs=torch.tensor([0.0]),
                        mean_token_log_prob=0.0,
                        shape_log_prob=candidate.log_prob,
                        source=candidate.source,
                    )
                ]

            model._sample_with_scores = types.MethodType(fake_sample_with_scores, model)
            model.sample_candidates = types.MethodType(fake_sample_candidates, model)

            metrics = evaluate_structured(
                model,
                dataset,
                "cpu",
                limit=1,
                sample_steps=1,
                infer_shape=True,
                shape_top_k=5,
            )

        self.assertNotIn((2, 2), recorded_shapes)
        self.assertEqual(metrics["shape_topk_hit"], 0.0)
        self.assertEqual(metrics["shape_exact"], 0.0)

    def test_inferred_shape_eval_can_select_shape_head_candidate(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4, 4], [4, 4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            dataset = ArcStructuredDataset([path])
            dataset.examples = [dataset.examples[-1]]
            model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)

            def fake_predict_shapes(self, **kwargs):
                return [[ShapeCandidate(2, 2, -0.2, "shape_head")]]

            def fake_sample_with_scores(self, target_rows, **kwargs):
                return (
                    torch.zeros_like(target_rows),
                    torch.zeros_like(target_rows, dtype=torch.float),
                )

            def fake_sample_candidates(self, **kwargs):
                candidates = []
                for shape in kwargs["candidate_shapes"]:
                    if (shape.height, shape.width) == (2, 2):
                        candidates.append(
                            StructuredCandidate(
                                height=2,
                                width=2,
                                sample=torch.tensor([4, 4, 4, 4]),
                                token_log_probs=torch.zeros(4),
                                mean_token_log_prob=0.0,
                                shape_log_prob=shape.log_prob,
                                source=shape.source,
                            )
                        )
                    else:
                        candidates.append(
                            StructuredCandidate(
                                height=shape.height,
                                width=shape.width,
                                sample=torch.zeros(shape.height * shape.width, dtype=torch.long),
                                token_log_probs=torch.full((shape.height * shape.width,), -10.0),
                                mean_token_log_prob=-10.0,
                                shape_log_prob=shape.log_prob,
                                source=shape.source,
                            )
                        )
                return candidates

            model.predict_shapes = types.MethodType(fake_predict_shapes, model)
            model._sample_with_scores = types.MethodType(fake_sample_with_scores, model)
            model.sample_candidates = types.MethodType(fake_sample_candidates, model)

            metrics = evaluate_structured(
                model,
                dataset,
                "cpu",
                limit=1,
                sample_steps=1,
                infer_shape=True,
                shape_top_k=3,
            )

        self.assertEqual(metrics["exact"], 1.0)
        self.assertEqual(metrics["shape_exact"], 1.0)
        self.assertEqual(metrics["shape_topk_hit"], 1.0)

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

    def test_structured_eval_writes_report_and_debug_json(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            report_path = Path(tmp) / "report.json"
            debug_dir = Path(tmp) / "debug"
            path.write_text(json.dumps(task), encoding="utf-8")
            dataset = ArcStructuredDataset([path])
            model = ArcOutputDiffusion(dim=32, max_grid_size=30, max_examples=8, num_heads=4)

            metrics = evaluate_structured(
                model,
                dataset,
                "cpu",
                limit=1,
                sample_steps=2,
                eval_report=report_path,
                debug_dir=debug_dir,
                debug_limit=1,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            debug_files = list(debug_dir.glob("*.json"))
            debug_payload = json.loads(debug_files[0].read_text(encoding="utf-8"))

        self.assertIn("cell_acc", metrics)
        self.assertIn("summary", report)
        self.assertIn("examples", report)
        self.assertEqual(len(debug_files), 1)
        self.assertIn("shape_exact", metrics)
        self.assertIn("predicted_shape", debug_payload)
        self.assertIn("trajectory", debug_payload)
        self.assertIn("confidence_grid", debug_payload)

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
        candidates = iter(
            [
                (torch.tensor([[3]]), torch.tensor([[-0.4]])),
                (torch.tensor([[4]]), torch.tensor([[-0.2]])),
                (torch.tensor([[3]]), torch.tensor([[-0.6]])),
                (torch.tensor([[4]]), torch.tensor([[-0.9]])),
            ]
        )

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

        candidates = iter(
            [
                (torch.tensor([[3]]), torch.tensor([[-0.5]])),
                (torch.tensor([[4]]), torch.tensor([[-0.5]])),
            ]
        )
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
