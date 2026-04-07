import torch

from src.models.model_factory import create_model


def test_model_creation():
    model = create_model("cnn", num_classes=2, dropout_rate=0.5)
    assert model is not None


def test_forward_pass():
    model = create_model("cnn", num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output is not None


def test_output_shape():
    model = create_model("cnn", num_classes=2)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 2)


def test_conv_feature_extraction():
    model = create_model("cnn", num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    model(dummy_input)
    features = model.get_conv_features()
    assert features is not None


def test_gradients_hook():
    model = create_model("cnn", num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    output = model(dummy_input)
    loss = output.mean()
    loss.backward()
    grads = model.get_gradients()
    assert grads is not None


def test_single_channel_model_forward_pass():
    model = create_model("cnn", num_classes=2, input_channels=1)
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)


if __name__ == "__main__":
    test_model_creation()
    test_forward_pass()
    test_output_shape()
    test_conv_feature_extraction()
    test_gradients_hook()

    print("===== Testing Model Definition =====")
    print("All Step 7 tests passed successfully.")