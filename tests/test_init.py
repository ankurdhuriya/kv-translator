import torch

def test_uv_environment_tensor_shapes():
    # Basic math sanity check for model tensor dimensions
    mock_tensor = torch.randn(1, 128, 8, 128)
    assert mock_tensor.shape == (1, 128, 8, 128)
