from tarp.services.training.context import TrainerContext

class Callback:
    def on_epoch_end(self, context: TrainerContext) -> None:
        pass

    def on_training_end(self, context: TrainerContext) -> None:
        pass

    def on_training_start(self, context: TrainerContext) -> None:
        pass

    def on_epoch_start(self, context: TrainerContext) -> None:
        pass