from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from anomalib.models.cflow.utils import get_logp, positional_encoding_2d
from anomalib.models.cflow import CflowLightning
from .dg_module import DgLightning

__all__ = ["CflowMmdLightning"]


class CflowMmdLightning(CflowLightning, DgLightning):
    def __init__(
            self,
            hparams: DictConfig
    ):
        CflowLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)

    def training_step(self, batch: dict, *args, **kwargs):
        del args, kwargs  # These variables are not used.

        opt = self.optimizers()
        self.model.encoder.eval()

        images = batch["image"]
        env_labels = self.get_env_labels(batch)
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            perm = torch.randperm(embedding_length)  # BHW
            decoder = self.model.decoders[layer_idx].to(images.device)

            fiber_batches = embedding_length // self.model.fiber_batch_size  # number of fiber batches
            assert fiber_batches > 0, "Make sure we have enough fibers, otherwise decrease N or batch-size!"

            for batch_num in range(fiber_batches):  # per-fiber processing
                opt.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size, (batch_num + 1) * self.model.fiber_batch_size
                    )
                else:  # When non-full batch is encountered batch_num * N will go out of bounds
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                # get random vectors
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                #
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob)
                loss += self.alpha * self.mmd_loss(env_labels, p_u, normalize_per_channel=False,
                                                   sum_of_distances=False, sigmas=self.sigmas)
                self.manual_backward(loss.mean())
                opt.step()
                avg_loss += loss.sum()

        self.log("train_loss", avg_loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": avg_loss}
