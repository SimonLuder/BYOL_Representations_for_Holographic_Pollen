import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from byol_poleno.model import SelfSupervisedLearner


class LITSSLModel(pl.LightningModule):
    def __init__(self, backbone, objective, image_size, image_channels, hidden_layer="avgpool", projection_size=256, projection_hidden_size=4096, 
                 augment_fn=torch.nn.Identity(), augment_fn2=None, moving_average_decay=0.99, use_momentum=True, lr=3e-4, val_knn=False,
                 ):
        super().__init__()
        self.model = SelfSupervisedLearner(
            backbone,
            objective = objective,
            image_size=image_size,
            image_channels=image_channels,
            hidden_layer=hidden_layer,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
            augment_fn=augment_fn,
            augment_fn2=augment_fn2, 
            moving_average_decay=moving_average_decay,
            use_momentum=use_momentum,
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        )
        self.lr = lr
        self.best_val_loss = float("inf")
        self.val_embeddings = []
        self.val_knn = val_knn
        self.val_knn_labels = []
        self.val_knn_labels_event = []
        self.val_pair_sims = []


    def training_step(self, batch, batch_idx):
        (im1, im2), _, _ = unpack_batch(batch)
        loss = self.model(x1=im1, x2=im2)

        local_bs = im1.size(0)

        # Log loss and training samples seen
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=local_bs)

        # Log embedding batch std every 100 steps
        if self.global_step % 100 == 0:
            emb1, emb2 = self.get_embbedings(im1, im2)
            emb_std = self.calc_embedding_std(emb1, emb2)
            self.log(
                "train_emb_std",
                emb_std,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=local_bs,
            )

        return loss


    def validation_step(self, batch, batch_idx):
        (im1, im2), (label1, label2), _ = unpack_batch(batch)
        loss = self.model(x1=im1, x2=im2)

        if label2 is None:
            label2 = label1

        # Store embeddings for epoch-end stats
        emb1, emb2 = self.get_embbedings(im1, im2)
        emb = torch.cat([emb1, emb2], dim=0)
        self.val_embeddings.append(emb.detach().cpu())

        pair_sim = self.pairwise_cosine_similarity(emb1, emb2)
        self.val_pair_sims.append(pair_sim.detach().cpu())

        # Class
        if self.val_knn == True and label1.get("class") is not None:
            labels = torch.cat([label1.get("class"), label2.get("class")], dim=0)
            self.val_knn_labels.append(labels.detach().cpu())

        # Event
        if self.val_knn == True and label1.get("event") is not None:
            labels = torch.cat([label1.get("event"), label2.get("event")], dim=0)
            self.val_knn_labels_event.append(labels.detach().cpu())

        local_bs = im1.size(0)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=local_bs)

        return loss


    def on_validation_epoch_end(self):

        if not self.val_embeddings:
            return
        
        # Event-wise similarity
        self.log_pairwise_sim()

        # Local embeddings: (N, D)
        emb = torch.cat(self.val_embeddings, dim=0).to(self.device)

        if self.val_knn:

            # Gather across GPUs
            emb = self.all_gather(emb)
            
            # DDP case: (world_size, N, D) -> (world_size * N, D)
            if emb.dim() == 3:
                emb = emb.flatten(0, 1)

            # Class
            if self.val_knn_labels:
                
                lbl = torch.cat(self.val_knn_labels, dim=0).to(self.device)         # Local labels: (N,)
                lbl = self.all_gather(lbl)                                          # Gather across GPUs
                if lbl.dim() > 1:                                                   # DDP case: (world_size, N) -> (world_size * N,)
                    lbl = lbl.flatten()

                assert emb.dim() == 2, emb.shape                                    # Sanity checks
                assert lbl.dim() == 1, lbl.shape
                assert emb.shape[0] == lbl.shape[0]

                if self.trainer.is_global_zero:
                    knn_acc = self.knn_accuracy(emb, lbl, k=10)
                    self.log("val_knn_acc_epoch", knn_acc, on_epoch=True)

            # Event
            if self.val_knn_labels_event:
                lble = torch.cat(self.val_knn_labels_event, dim=0).to(self.device)  # Local labels: (N,)
                lble = self.all_gather(lble)                                        # Gather across GPUs
                if lble.dim() > 1:                                                  # DDP case: (world_size, N) -> (world_size * N,)
                    lble = lble.flatten()

                assert emb.dim() == 2, emb.shape                                    # Sanity checks
                assert lble.dim() == 1, lble.shape
                assert emb.shape[0] == lble.shape[0]

                if self.trainer.is_global_zero:
                    knn_acc_e = self.knn_accuracy(emb, lble, k=1)
                    mrr_e = self.mean_reciprocal_rank(emb, lble)

                    self.log("val_event_knn_acc_epoch", knn_acc_e, on_epoch=True)
                    self.log("val_event_mrr_epoch", mrr_e, on_epoch=True)

        # Embedding stats (local stats are fine here)
        embeddings = torch.cat(self.val_embeddings, dim=0)
        emb_std = embeddings.std(dim=0).mean()
        self.log("val_emb_std_epoch", emb_std, sync_dist=True)

        # Clear buffers
        self.val_embeddings.clear()
        self.val_knn_labels.clear()
        self.val_knn_labels_event.clear()
        self.val_pair_sims.clear()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


    def get_embbedings(self, x1, x2):
        with torch.no_grad():
            emb1, emb2 = self.model(
                x1=x1,
                x2=x2,
                return_embedding=True,
                return_projection=False,
                validation=True,
            )
            
        return emb1, emb2


    def calc_embedding_std(self, emb1, emb2):
        emb = torch.cat([emb1, emb2], dim=0)
        emb_std = emb.std(dim=0).mean()
        return emb_std

    @staticmethod
    @torch.no_grad()
    def pairwise_cosine_similarity(emb1, emb2):
        # Normalize for cosine similarity
        emb1_n = F.normalize(emb1, dim=1)
        emb2_n = F.normalize(emb2, dim=1)

        # Same-event similarity (index-aligned)
        pair_sim = (emb1_n * emb2_n).sum(dim=1)  # (B,)
        return pair_sim
    

    @staticmethod
    @torch.no_grad()
    def knn_accuracy(emb, labels, k=10):
        emb = F.normalize(emb, dim=1)
        sim = emb @ emb.T
        sim.fill_diagonal_(-float("inf"))

        topk = sim.topk(k=k, dim=1).indices
        preds, _ = torch.mode(labels[topk], dim=1)

        return (preds == labels).float().mean()
    

    def log_pairwise_sim(self):
        if not self.val_pair_sims:
            return

        pair_sim = torch.cat(self.val_pair_sims, dim=0).to(self.device)

        # Gather across GPUs
        pair_sim = self.all_gather(pair_sim)

        # (world_size, N) → (world_size * N,)
        if pair_sim.dim() > 1:
            pair_sim = pair_sim.flatten()

        avg_sim = pair_sim.mean()
        std_sim = pair_sim.std()

        if self.trainer.is_global_zero:
            self.log("val_same_event_similarity", avg_sim, on_epoch=True)
            self.log("val_same_event_similarity_std", std_sim, on_epoch=True)
                

    @staticmethod
    @torch.no_grad()
    def mean_reciprocal_rank(emb, labels):
        """
        emb: (N, D) normalized embeddings
        labels: (N,) integer labels
        """
        emb = F.normalize(emb, dim=1)

        # Cosine similarity matrix
        sim = emb @ emb.T
        sim.fill_diagonal_(-float("inf"))

        # Sort indices by similarity (descending)
        ranked_indices = sim.argsort(dim=1, descending=True)

        # Gather labels according to ranking
        ranked_labels = labels[ranked_indices]  # (N, N)

        # Find first correct match per query
        correct = ranked_labels == labels.unsqueeze(1)

        # Index of first True (rank is 1-based)
        ranks = correct.float().argmax(dim=1) + 1

        # Handle cases with no correct match (should not happen, but safe)
        valid = correct.any(dim=1)
        reciprocal_ranks = torch.zeros_like(ranks, dtype=torch.float)
        reciprocal_ranks[valid] = 1.0 / ranks[valid].float()

        return reciprocal_ranks.mean()
    

def unpack_batch(batch):
    """
    Supports:
      ((im1, im2), (label1, label2), (fname1, fname2))
      (im, label, fname)

    Returns:
      (x1, x2), (y1, y2), (z1, z2)
      (x2, y2 and z2, are None for single-image batches)
    """
    images, labels, fnames = batch

    if isinstance(images, (tuple, list)):
        x1, x2 = images
        y1, y2 = labels
        z1, z2 = fnames
    else:
        x1, x2 = images, None
        y1, y2 = labels, None
        z1, z2 = fnames, None

    return (x1, x2), (y1, y2), (z1, z2)