"""CPU regression tests for the dataset-to-collator boundary.

Only the production definitions under test are loaded through AST, so importing
these tests does not require model checkpoints or optional tokenizer packages.
NumPy, pandas, SciPy and PyTorch are real dependencies. The loader substitutes
only preprocessing (the fixture is already a prepared DataFrame/token table).

Run: python -m unittest discover -s tests -p 'test_scfoundation_dataset.py' -v
Set UNICELL_TEST_ROOT to another checkout to reproduce the pre-fix failure.
"""

import ast
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("UNICELL_TEST_ROOT", Path(__file__).resolve().parents[1]))


def source_node(relative_path, *names):
    """Read an unchanged definition from source without importing its module."""
    node = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    for name in names:
        node = next(child for child in node.body if getattr(child, "name", None) == name)
    return node


def load_definitions(relative_path, names, namespace):
    tree = ast.Module(
        body=[source_node(relative_path, *name.split(".")) for name in names],
        type_ignores=[],
    )
    exec(compile(tree, str(ROOT / relative_path), "exec"), namespace)


NAMESPACE = {
    "torch": torch,
    "np": np,
    "re": re,
    "Dataset": Dataset,
    "issparse": sparse.issparse,
    "load_scf_data": lambda adata, vocab: adata.X,
    "load_gf_data": lambda adata: adata.X,
    "load_gpt_data": lambda adata, vocab, args: adata.X,
}
load_definitions("unicell/dataset.py", ["HMCNDataset"], NAMESPACE)
load_definitions(
    "unicell/repo/geneformer/in_silico_perturber.py",
    ["get_model_input_size", "pad_tensor", "pad_tensor_list"],
    NAMESPACE,
)
load_definitions("unicell/trainer.py", ["UnicellTrainer.collate_fn"], NAMESPACE)
load_definitions("unicell/anno_predict.py", ["collate_fn_with_args"], NAMESPACE)
HMCNDataset = NAMESPACE["HMCNDataset"]
TRAIN_COLLATE = NAMESPACE["collate_fn"]
PREDICT_COLLATE = NAMESPACE["collate_fn_with_args"]


def stack_from_source(relative_path, names, batch_data):
    """Execute the actual stack assignment used by one of the three callers."""
    scope = source_node(relative_path, *names)
    expected = ast.dump(ast.parse("torch.stack(batch_data)", mode="eval").body)
    assignments = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(child, ast.Call) and ast.dump(child) == expected
            for child in ast.walk(node.value)
        )
    ]
    if len(assignments) != 1:
        raise AssertionError("Expected exactly one scFoundation stack assignment")
    namespace = {
        "torch": torch,
        "batch_data": batch_data,
        "self": SimpleNamespace(device=torch.device("cpu")),
        "device": torch.device("cpu"),
    }
    tree = ast.Module(body=assignments, type_ignores=[])
    exec(compile(tree, str(ROOT / relative_path), "exec"), namespace)
    return namespace["batch_data"]


class TokenRows:
    """Minimal prepared GeneFormer table implementing its select contract."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows),)

    def select(self, indices):
        return [self.rows[index] for index in indices]


def prepared_dataset(data, input_type="scFoundation", batch=False, auxiliary=True):
    scdata = SimpleNamespace(
        adata=SimpleNamespace(
            X=data,
            obs=pd.DataFrame({"celltype": ["type-z", "type-a"]}),
        ),
        cell_type_index=np.array([7, 3]),
        cell_type_key="celltype",
        batch_key="batch" if batch else None,
        batch_index=np.array([9, 4]),
        llm_vocab={},
        llm_args={},
    )
    if auxiliary:
        scdata.tissue_index = np.array([2, 5])
        scdata.species_index = np.array([1, 0])
    return HMCNDataset(scdata, input_type)


def aligned_frame(dtype=np.float32):
    # Deliberately reverse the names so sorting columns would change gene order.
    values = np.arange(2 * 19264, dtype=dtype).reshape(2, 19264) / dtype(8)
    return pd.DataFrame(
        values,
        columns=["gene_%05d" % i for i in range(19263, -1, -1)],
        index=["cell-z", "cell-a"],
    )


def collators(dataset):
    model = SimpleNamespace(
        geneformer=SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(position_embeddings=torch.nn.Embedding(8, 2))
            )
        ),
        llm_vocab={"<pad>": 0},
    )
    trainer = SimpleNamespace(
        input_type=dataset.input_type,
        dataset=dataset,
        _base_model=lambda: model,
    )
    return (
        lambda batch: TRAIN_COLLATE(trainer, batch),
        PREDICT_COLLATE(dataset.input_type, model, dataset),
    )


class ScFoundationDatasetTests(unittest.TestCase):
    def test_float32_and_float64_rows_preserve_values_width_and_gene_order(self):
        for dtype, torch_dtype in ((np.float32, torch.float32), (np.float64, torch.float64)):
            with self.subTest(dtype=dtype):
                frame = aligned_frame(dtype)
                dataset = prepared_dataset(frame)
                self.assertEqual(len(dataset), 2)
                for index in (0, 1):
                    row = dataset[index][0]
                    self.assertIsInstance(row, torch.Tensor)
                    self.assertEqual(row.dtype, torch_dtype)
                    self.assertEqual(tuple(row.shape), (19264,))
                    np.testing.assert_array_equal(row.numpy(), frame.iloc[index].to_numpy())

    def test_five_and_six_field_label_contracts(self):
        for batch in (False, True):
            for auxiliary in (False, True):
                with self.subTest(batch=batch, auxiliary=auxiliary):
                    dataset = prepared_dataset(aligned_frame(), batch=batch, auxiliary=auxiliary)
                    item = dataset[1]
                    self.assertEqual(len(item), 6 if batch else 5)
                    self.assertEqual(item[1].dtype, torch.long)
                    self.assertEqual(item[1].item(), 3)
                    self.assertEqual(item[2], "type-a")
                    self.assertEqual(item[3:5], (5, 0) if auxiliary else (-1, -1))
                    if batch:
                        self.assertEqual(item[5].dtype, torch.long)
                        self.assertEqual(item[5].item(), 4)

    def test_returned_rows_own_storage_in_both_directions(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                frame = aligned_frame(dtype)
                dataset = prepared_dataset(frame)
                first = dataset[0][0]
                second = dataset[0][0]
                self.assertIsInstance(first, torch.Tensor)
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                first[0] = -123
                self.assertEqual(frame.iloc[0, 0], 0)
                self.assertEqual(second[0].item(), 0)
                original = second[1].item()
                frame.iloc[0, 1] = -456
                self.assertEqual(first[1].item(), original)
                self.assertEqual(second[1].item(), original)

    def test_tiny_positive_float64_values_keep_the_expression_mask(self):
        frame = aligned_frame(np.float64)
        frame.iloc[0, :4] = [0.0, 1e-50, -1e-50, 1.0000000000000002]
        row = prepared_dataset(frame)[0][0]
        self.assertIsInstance(row, torch.Tensor)
        self.assertEqual(row.dtype, torch.float64)
        np.testing.assert_array_equal(row.numpy() > 0, frame.iloc[0].to_numpy() > 0)
        self.assertEqual(row[1].item(), 1e-50)
        self.assertEqual(row[3].item(), 1.0000000000000002)

    def check_stack_caller(self, relative_path, names, prediction=False):
        for dtype in (np.float32, np.float64):
            for batch in (False, True):
                with self.subTest(dtype=dtype, batch=batch):
                    frame = aligned_frame(dtype)
                    dataset = prepared_dataset(frame, batch=batch)
                    collate = collators(dataset)[int(prediction)]
                    loader = DataLoader(dataset, batch_size=2, sampler=[1, 0], collate_fn=collate)
                    packed = next(iter(loader))
                    stacked = stack_from_source(relative_path, names, packed[0])
                    self.assertEqual(tuple(stacked.shape), (2, 19264))
                    np.testing.assert_array_equal(stacked.numpy(), frame.iloc[[1, 0]].to_numpy())
                    self.assertEqual(stacked.dtype, torch.float32 if dtype == np.float32 else torch.float64)
                    self.assertEqual(packed[1].tolist(), [3, 7])
                    self.assertEqual(packed[2], ("type-a", "type-z"))
                    self.assertEqual(packed[3:5], ([5, 2], [0, 1]))
                    self.assertEqual(len(packed), 6 if batch else 5)
                    if batch:
                        self.assertEqual(packed[5].tolist(), [4, 9])

    def test_training_collator_and_stack(self):
        self.check_stack_caller("unicell/trainer.py", ("UnicellTrainer", "train_one_epoch"))

    def test_validation_collator_and_stack(self):
        self.check_stack_caller("unicell/trainer.py", ("UnicellTrainer", "predict"))

    def test_prediction_collator_and_stack(self):
        self.check_stack_caller("unicell/anno_predict.py", ("unicell_predict",), prediction=True)

    def test_expr_still_returns_indices_and_batches_dense_or_sparse_values(self):
        values = np.array([[1.5, 0, 2], [0, 3.5, 4]], dtype=np.float64)
        for matrix in (values, sparse.csr_matrix(values)):
            dataset = prepared_dataset(matrix, input_type="expr", batch=True)
            self.assertEqual(dataset[1][0], 1)
            for collate in collators(dataset):
                packed = collate([dataset[1], dataset[0]])
                self.assertEqual(packed[0].dtype, torch.float32)
                np.testing.assert_array_equal(packed[0].numpy(), values[[1, 0]].astype(np.float32))
                self.assertEqual(packed[1].tolist(), [3, 7])
                self.assertEqual(packed[5].tolist(), [4, 9])

    def test_geneformer_still_selects_records_and_pads_token_batches(self):
        records = [{"input_ids": [8, 4], "length": 2}, {"input_ids": [7], "length": 1}]
        dataset = prepared_dataset(TokenRows(records), input_type="GeneFormer")
        self.assertEqual(len(dataset), 2)
        self.assertIs(dataset[1][0], records[1])
        for collate in collators(dataset):
            packed = collate([dataset[1], dataset[0]])
            self.assertEqual(packed[0]["input_ids"].tolist(), [[7, 0], [8, 4]])
            self.assertEqual(packed[0]["input_ids"].dtype, torch.long)
            self.assertEqual(packed[0]["length"].tolist(), [1, 2])
            self.assertEqual(packed[1].tolist(), [3, 7])

    def test_scgpt_still_returns_token_dicts_and_passes_them_through_collation(self):
        tokens = {
            "input_ids": torch.tensor([[1, 5, 0], [1, 9, 2]]),
            "values": torch.tensor([[-2, 0.5, -2], [-2, 2.5, 3.5]]),
        }
        dataset = prepared_dataset(tokens, input_type="scGPT", batch=True)
        self.assertEqual(len(dataset), 2)
        items = [dataset[1], dataset[0]]
        for key, value in tokens.items():
            self.assertTrue(torch.equal(items[0][0][key], value[1]))
        for collate in collators(dataset):
            packed = collate(items)
            self.assertIs(packed[0][0], items[0][0])
            for key, value in tokens.items():
                self.assertTrue(torch.equal(torch.stack([row[key] for row in packed[0]]), value[[1, 0]]))
            self.assertEqual(packed[1].tolist(), [3, 7])
            self.assertEqual(packed[5].tolist(), [4, 9])


if __name__ == "__main__":
    unittest.main()
