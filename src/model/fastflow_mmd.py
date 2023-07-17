from anomalib.models.fastflow import FastflowLightning
from .dg_module import DgLightning


class FastflowMmdLightning(FastflowLightning, DgLightning):
    def __init__(self, hparams):
        FastflowLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)

    def training_step(self, batch, *args, **kwargs):
        """Forward-pass input and return the loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Input batch
            _batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        hidden_variables, jacobians = self.model(batch["image"])
        env_labels = self.get_env_labels(batch)
        loss = self.loss(hidden_variables, jacobians)
        for hidden_variable in hidden_variables:
            loss += self.alpha * self.mmd_loss(env_labels, hidden_variable, normalize_per_channel=False,
                                               sum_of_distances=False, sigmas=self.sigmas)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
