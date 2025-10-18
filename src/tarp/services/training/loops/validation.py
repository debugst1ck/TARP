from torch.utils.data import DataLoader
from tarp.services.training.loops import Loop
from tqdm import tqdm
import torch

class ValidationLoop(Loop):
    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.context.model.eval()
        total_loss = 0.0
        all_expected, all_predictions = [], []
        loop = tqdm(
            dataloader,
            desc=f"Validation {epoch+1}/{self.context.epochs}",
            unit="batch",
            colour="red",
        )
        with torch.no_grad():
            for batch in loop:
                with torch.amp.autocast(
                    device_type=self.context.device.type,
                    enabled=self.context.use_amp,
                ):
                    loss, predictions, expected = self.iteration(batch)
                total_loss += loss.item()
                if predictions is not None:
                    all_predictions.append(predictions)
                if expected is not None:
                    all_expected.append(expected)
                loop.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = total_loss / len(dataloader)
        metrics = self.evaluation(all_predictions, all_expected)
        metrics["validation_loss"] = average_loss
        return metrics