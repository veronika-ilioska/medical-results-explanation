"""CPU-only batching regressions using fake model/tokenizer outputs.

These check generation plumbing and CSV checkpoints without model downloads,
CUDA, or the optional torch/transformers/peft dependencies.
"""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Tensor(np.ndarray):
    @property
    def device(self):
        return "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self


def tensor(values, dtype=None, device=None):
    return np.asarray(values, dtype=dtype).view(Tensor)


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    pad_token = "<pad>"
    padding_side = "left"

    def __call__(self, prompts, **kwargs):
        assert kwargs["padding"] is True
        assert kwargs["return_attention_mask"] is True
        assert kwargs["return_tensors"] == "pt"
        width = max(map(len, prompts))
        ids = [[0] * (width - len(prompt)) + list(map(ord, prompt)) for prompt in prompts]
        return Batch(input_ids=tensor(ids), attention_mask=tensor(np.asarray(ids) != 0))

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["add_generation_prompt"] is True
        self.conversations = conversations
        prompts = []
        for messages in conversations:
            content = messages[-1]["content"]
            prompts.append(content if isinstance(content, str) else content[0]["text"])
        return self(prompts, **kwargs)

    def decode(self, ids, skip_special_tokens):
        return " ".join(str(int(value)) for value in ids if not skip_special_tokens or value not in (0, 1, 2))

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row, **kwargs) for row in ids]


class Processor(Tokenizer):
    def __init__(self):
        self.tokenizer = Tokenizer()


class Model:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        # Different response lengths, including EOS and padding after completion.
        suffixes = [[201, 2, 0], [202, 203, 2]][:len(kwargs["input_ids"])]
        return tensor(np.concatenate([kwargs["input_ids"], suffixes], axis=1))


def load_script(name):
    torch = MagicMock()
    torch.inference_mode = nullcontext
    torch.equal = np.array_equal
    torch.tensor = tensor
    torch.bool = np.bool_
    transformers = MagicMock()
    transformers.LogitsProcessor = object
    transformers.LogitsProcessorList = list
    spec = importlib.util.spec_from_file_location(
        f"{name}_generation_under_test", ROOT / "scripts" / name / "generate_outputs.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": MagicMock()}):
        spec.loader.exec_module(module)
    return module


SCRIPTS = {name: load_script(name) for name in ("llama", "medgemma", "tablellm")}


def medgemma_args(**overrides):
    values = dict(
        system_mode="none", max_new_tokens=8, min_new_tokens=0,
        do_sample=False, repetition_penalty=1.1, allow_pad_generation=False,
        debug_generations=False, temperature=0.7, top_p=0.9,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class GenerationBatchingTests(unittest.TestCase):
    def test_mixed_lengths_use_one_model_call_and_decode_only_responses(self):
        for name, module in SCRIPTS.items():
            with self.subTest(model=name):
                frontend = Processor() if name == "medgemma" else Tokenizer()
                model = Model()
                options = medgemma_args() if name == "medgemma" else 8
                self.assertEqual(
                    module.generate_batch(frontend, model, ["x", "abc"], options),
                    ["201", "202 203"],
                )
                self.assertEqual(len(model.calls), 1)
                call = model.calls[0]
                np.testing.assert_array_equal(call["input_ids"], [[0, 0, 120], [97, 98, 99]])
                np.testing.assert_array_equal(call["attention_mask"], [[0, 0, 1], [1, 1, 1]])
                self.assertEqual(call["max_new_tokens"], 8)
                self.assertFalse(call["do_sample"])
                if name == "llama":
                    self.assertEqual(frontend.conversations[0][0]["content"], module.SYSTEM_PROMPT)

    def test_empty_and_single_prompt_batches(self):
        for name, module in SCRIPTS.items():
            with self.subTest(model=name):
                frontend = Processor() if name == "medgemma" else Tokenizer()
                model = Model()
                options = medgemma_args() if name == "medgemma" else 8
                self.assertEqual(module.generate_batch(frontend, model, [], options), [])
                self.assertEqual(model.calls, [])
                self.assertEqual(module.generate(frontend, model, "x", options), "201")
                self.assertEqual(len(model.calls), 1)

    def test_loaders_enable_left_padding(self):
        args = SimpleNamespace(model="local", local_files_only=True, quantization="none", adapter=None)
        for name, module in SCRIPTS.items():
            with self.subTest(model=name):
                frontend = Processor() if name == "medgemma" else Tokenizer()
                tokenizer = frontend.tokenizer if name == "medgemma" else frontend
                tokenizer.padding_side = "right"
                factory = module.AutoProcessor if name == "medgemma" else module.AutoTokenizer
                loader = module.load_model if name == "llama" else module.load
                with patch.object(factory, "from_pretrained", return_value=frontend):
                    loader(args)
                self.assertEqual(tokenizer.padding_side, "left")

    def test_medgemma_preserves_options_debug_and_token_suppression(self):
        module = SCRIPTS["medgemma"]
        for mode in ("none", "prepend", "separate"):
            with self.subTest(system_mode=mode):
                frontend, model = Processor(), Model()
                args = medgemma_args(system_mode=mode, do_sample=True, min_new_tokens=2, debug_generations=True)
                captured = io.StringIO()
                with redirect_stdout(captured):
                    results = module.generate_batch(frontend, model, ["x", "abc"], args)
                self.assertEqual(results, ["201", "202 203"])
                self.assertEqual(frontend.conversations, [module.build_messages(p, mode) for p in ["x", "abc"]])
                call = model.calls[0]
                for key in ("temperature", "top_p", "min_new_tokens", "repetition_penalty"):
                    self.assertEqual(call[key], getattr(args, key))
                scores = tensor(np.zeros((2, 5)))
                suppressed = call["logits_processor"][0](call["input_ids"], scores)
                self.assertTrue(np.isneginf(suppressed[:, 0]).all())
                self.assertIn("Batch item 2/2", captured.getvalue())
                self.assertEqual(call["suppress_tokens"], [0])
        model = Model()
        module.generate_batch(Processor(), model, ["x"], medgemma_args(allow_pad_generation=True))
        self.assertNotIn("suppress_tokens", model.calls[0])

    def test_batch_size_cli_validation_and_default(self):
        for name, module in SCRIPTS.items():
            parse = module.parse_args if name == "llama" else module.arguments
            argv = ["generate_outputs.py", "--input", __file__, "--output", "unused.csv", "--prediction-column", "prediction"]
            for size in (None, "4", "0", "-2", "abc"):
                with self.subTest(model=name, size=size), patch.object(sys, "argv", argv + ([] if size is None else ["--batch-size", size])):
                    if size in ("0", "-2", "abc"):
                        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                            parse()
                        self.assertEqual(error.exception.code, 2)
                    else:
                        self.assertEqual(parse().batch_size, 1 if size is None else 4)

    def test_interruption_resume_row_order_and_partial_last_batch(self):
        for name, module in SCRIPTS.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.csv"
                output = Path(directory) / "predictions.csv"
                pd.DataFrame({
                    "prompt": [f"BLOOD TEST RESULTS:\n- Test{i}: {i} units [normal]" for i in range(6)],
                    "generated_text": ["reference"] * 6,
                    "prediction": ["", "existing", "", "", "", ""],
                    "rendered": ["", "existing prompt", "", "", "", ""],
                }).to_csv(source, index=False)
                args = SimpleNamespace(
                    input=source, output=output, overwrite=False, max_rows=None,
                    prediction_column="prediction", prompt_output_column="rendered",
                    batch_size=2, max_new_tokens=8,
                )
                parse_name = "parse_args" if name == "llama" else "arguments"
                load_name = "load_model" if name == "llama" else "load"
                with patch.object(module, parse_name, return_value=args), patch.object(module, load_name) as loader, redirect_stdout(io.StringIO()):
                    loader.return_value = (object(), object())
                    with patch.object(module, "generate_batch", side_effect=[["answer0", "answer2"], RuntimeError("interrupted")]):
                        with self.assertRaisesRegex(RuntimeError, "interrupted"):
                            module.main()
                    saved = pd.read_csv(output).fillna("")
                    self.assertEqual(saved.prediction.tolist(), ["answer0", "existing", "answer2", "", "", ""])
                    # Resume with a different batch size and an incomplete final batch.
                    args.batch_size = 4
                    with patch.object(module, "generate_batch", return_value=["answer3", "answer4", "answer5"]) as generate:
                        module.main()
                    self.assertEqual(len(generate.call_args.args[2]), 3)
                    saved = pd.read_csv(output).fillna("")
                    self.assertEqual(saved.prediction.tolist(), ["answer0", "existing", "answer2", "answer3", "answer4", "answer5"])
                    self.assertEqual(saved.loc[1, "rendered"], "existing prompt")
                    for index in (0, 2, 3, 4, 5):
                        self.assertIn(f"Test{index}", saved.loc[index, "rendered"])
                    loader.reset_mock()
                    module.main()
                    loader.assert_not_called()
                    self.assertFalse(output.with_suffix(".csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
