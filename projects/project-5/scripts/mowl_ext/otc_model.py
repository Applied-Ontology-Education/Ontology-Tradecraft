import logging

import numpy as np
import torch as th
from mowl.models import ELEmbeddings
from tqdm import trange

from util.logger_config import config

logger = logging.getLogger(__name__)

config(logger)


class OtcModel(ELEmbeddings):
    def train(self, epochs=None, validate_every=1):
        logger.warning(
            'You are using the default training method. If you want to use a cutomized training method (e.g., different negative sampling, etc.), please reimplement the train method in a subclass.')

        # Expose losses for external selection/tuning logic
        if not hasattr(self, "best_validation_loss"):
            self.best_validation_loss = None
        if not hasattr(self, "last_validation_loss"):
            self.last_validation_loss = None
        if not hasattr(self, "last_train_loss"):
            self.last_train_loss = None

        points_per_dataset = {k: len(v) for k, v in self.training_datasets.items()}
        string = "Training datasets: \n"
        for k, v in points_per_dataset.items():
            string += f"\t{k}: {v}\n"

        logger.info(string)

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        all_classes_ids = list(self.class_index_dict.values())
        all_inds_ids = list(self.individual_index_dict.values())

        if epochs is None:
            epochs = self.epochs

        for epoch in trange(epochs):
            self.module.train()

            train_loss = 0
            loss = 0

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                loss += th.mean(self.module(gci_dataset[:], gci_name))
                if gci_name == "gci2":
                    idxs_for_negs = np.random.choice(all_classes_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))

                if gci_name == "object_property_assertion":
                    idxs_for_negs = np.random.choice(all_inds_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))

            loss += self.module.regularization_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            self.last_train_loss = float(train_loss)

            loss = 0

            if (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    with th.no_grad():
                        self.module.eval()
                        valid_loss = 0.0
                        contributing = 0
                        # Iterate over all available validation GCI datasets.
                        # Note: some gci_names in validation may be empty; skip those.
                        for gci_name, gci_dataset in self.validation_datasets.items():
                            try:
                                if len(gci_dataset) == 0:
                                    continue
                            except Exception:
                                # If dataset does not support len(), try slicing and checking shape
                                data_slice = gci_dataset[:]
                                if data_slice is None or (hasattr(data_slice, 'shape') and data_slice.shape[0] == 0):
                                    continue
                                # If we got here, restore reference
                                gci_dataset = data_slice

                            data = gci_dataset[:]
                            # Compute mean loss for this GCI on validation (positives only)
                            l = th.mean(self.module(data, gci_name))
                            valid_loss += l.detach().item()
                            contributing += 1

                        if contributing > 0:
                            # Average across present validation GCI datasets so it is
                            # more comparable to the aggregated training loss.
                            valid_loss = valid_loss / contributing
                            self.last_validation_loss = float(valid_loss)
                            if valid_loss < best_loss:
                                best_loss = valid_loss
                                self.best_validation_loss = float(best_loss)
                                th.save(self.module.state_dict(), self.model_filepath)
                            print(
                                f'Epoch {epoch + 1}: Train loss: {train_loss} Valid loss: {valid_loss} (n={contributing})')
                        else:
                            # No validation data available this epoch.
                            print(f'Epoch {epoch + 1}: Train loss: {train_loss} (no validation data)')
                else:
                    print(f'Epoch {epoch + 1}: Train loss: {train_loss}')
