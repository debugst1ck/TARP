from torch.utils.data import DataLoader
from tarp.services.training.loops import Loop
from tqdm import tqdm
import torch
from torch import nn

class TrainingLoop(Loop):
    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.context.model.train()
        total_loss = 0.0
        loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{self.context.epochs}",
            unit="batch",
            colour="green",
        )
        for step, batch in enumerate(loop):
            with torch.amp.autocast(
                device_type=self.context.device.type,
                enabled=self.context.use_amp,
            ):
                raw_loss, _, _ = self.iteration(batch)  # Loss, predictions, expected
                loss = raw_loss / self.context.accumulation_steps

            self.context.scaler.scale(loss).backward()
            # Gradient accumulation step
            accumulation_stop = (step + 1) % self.context.accumulation_steps == 0
            is_last_step = (step + 1) == len(dataloader)
            if accumulation_stop or is_last_step:
                # Unscale gradients and perform gradient clipping
                self.context.scaler.unscale_(self.context.optimizer)
                nn.utils.clip_grad_norm_(
                    self.context.model.parameters(),
                    self.context.gradient_clipping_threshold,
                )

                # Update parameters
                self.context.scaler.step(self.context.optimizer)  # Update parameters
                self.context.scaler.update()  # Update the scale for next iteration
                self.context.optimizer.zero_grad(
                    set_to_none=True
                )  # Reset gradients for next step

            total_loss += raw_loss.item()
            loop.set_postfix(loss=f"{raw_loss.item():.4f}")

        average_loss = total_loss / len(dataloader)
        return {"training_loss": average_loss}