import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.predictor import Predictor
from src.models.model_factory import create_model


def test_prediction_collection():

    device = torch.device("cpu")

    model = create_model("cnn", num_classes=2)

    predictor = Predictor(model=model, device=device)

    dummy_images = torch.randn(10, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (10,))

    dataset = TensorDataset(dummy_images, dummy_labels)

    dataloader = DataLoader(dataset, batch_size=2)

    results = predictor.collect_predictions(dataloader)

    true_labels = results["true_labels"]
    predicted_labels = results["predicted_labels"]
    probabilities = results["probabilities"]

    assert len(true_labels) == 10
    assert len(predicted_labels) == 10
    assert len(probabilities) == 10

    print("Prediction collection test passed.")
    print(f"Samples processed: {len(true_labels)}")


if __name__ == "__main__":
    test_prediction_collection()