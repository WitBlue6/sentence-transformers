from __future__ import annotations

from pathlib import Path

import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from sentence_transformers import CrossEncoder


@pytest.fixture
def local_checkpoint(tmp_path: Path) -> str:
    """Build a small sequence classifier and tokenizer without Hub downloads."""
    config = BertConfig(
        vocab_size=7,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
    )
    BertForSequenceClassification(config).save_pretrained(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab)).save_pretrained(tmp_path)
    return str(tmp_path)


def test_num_labels_does_not_change_shared_config(local_checkpoint: str) -> None:
    config_kwargs = {"output_attentions": False}
    model = CrossEncoder(
        local_checkpoint,
        num_labels=2,
        config_kwargs=config_kwargs,
        local_files_only=True,
        device="cpu",
    )
    assert model.config.num_labels == 2
    assert config_kwargs == {"output_attentions": False}


def test_max_length_does_not_change_shared_processor_config(local_checkpoint: str) -> None:
    processor_kwargs = {"model_max_length": 64}
    short_model = CrossEncoder(
        local_checkpoint,
        max_length=32,
        processor_kwargs=processor_kwargs,
        local_files_only=True,
        device="cpu",
    )
    assert short_model.max_seq_length == 32
    assert processor_kwargs == {"model_max_length": 64}
    original_model = CrossEncoder(
        local_checkpoint,
        processor_kwargs=processor_kwargs,
        local_files_only=True,
        device="cpu",
    )
    assert original_model.max_seq_length == 64


@pytest.mark.parametrize("config_key", ["config_kwargs", "config_args"])
def test_classifier_dropout_does_not_change_shared_config(local_checkpoint: str, config_key: str) -> None:
    config_kwargs = {"classifier_dropout": 0.1}
    model = CrossEncoder(
        local_checkpoint,
        classifier_dropout=0.25,
        **{config_key: config_kwargs},
        local_files_only=True,
        device="cpu",
    )
    assert model.config.classifier_dropout == 0.25
    assert config_kwargs == {"classifier_dropout": 0.1}


def test_classifier_dropout_accepts_explicit_none(local_checkpoint: str) -> None:
    model = CrossEncoder(
        local_checkpoint,
        classifier_dropout=0.25,
        config_kwargs=None,
        local_files_only=True,
        device="cpu",
    )
    assert model.config.classifier_dropout == 0.25
