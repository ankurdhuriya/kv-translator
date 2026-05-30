from unittest.mock import MagicMock, patch
import torch
from kv_translator.data_pipeline import KVDatasetGenerator


@patch("kv_translator.data_pipeline.AutoTokenizer")
@patch("kv_translator.data_pipeline.AutoModelForCausalLM")
def test_dataset_generator_mock_extraction(mock_model_cls, mock_tokenizer_cls):
    # Setup mock behaviors
    mock_tokenizer = MagicMock()
    mock_tokenizer_output = MagicMock()
    mock_tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer_output.to.return_value = mock_tokenizer_output
    mock_tokenizer.return_value = mock_tokenizer_output
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    mock_src_model = MagicMock()
    mock_tg_model = MagicMock()

    # Replicate native outputs fields
    mock_src_outputs = MagicMock()
    mock_src_outputs.past_key_values = ("src_kv_layer_1", "src_kv_layer_2")
    mock_src_model.return_value = mock_src_outputs

    mock_tg_outputs = MagicMock()
    mock_tg_outputs.past_key_values = ("tg_kv_layer_1", "tg_kv_layer_2")
    mock_tg_model.return_value = mock_tg_outputs

    # Configure class factory parameters
    mock_model_cls.from_pretrained.side_effect = [mock_src_model, mock_tg_model]

    # Instantiate generator under device isolation override
    generator = KVDatasetGenerator("mock_src", "mock_tg", device="cpu")
    src_kv, tg_kv = generator.extract_pair("Sample text data")

    assert src_kv == ("src_kv_layer_1", "src_kv_layer_2")
    assert tg_kv == ("tg_kv_layer_1", "tg_kv_layer_2")
