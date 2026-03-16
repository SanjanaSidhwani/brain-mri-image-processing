import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List

from src.models.model_factory import create_model


class Predictor:

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, device: torch.device):

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = create_model(
            architecture=checkpoint["args"]["architecture"],
            num_classes=checkpoint["args"]["num_classes"],
            dropout_rate=checkpoint["args"]["dropout_rate"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return Predictor(model=model, device=device)

    def collect_predictions(self, dataloader: DataLoader) -> Dict[str, List]:

        true_labels = []
        predicted_labels = []
        probabilities = []

        with torch.no_grad():

            for images, labels in dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                probs = F.softmax(outputs, dim=1)

                preds = torch.argmax(probs, dim=1)

                true_labels.extend(labels.cpu().numpy().tolist())
                predicted_labels.extend(preds.cpu().numpy().tolist())
                probabilities.extend(probs.cpu().numpy().tolist())

        return {
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "probabilities": probabilities,
        }