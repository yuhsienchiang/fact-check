import torch
from torch import Tensor as T
from torch import nn
from torch.utils.data import DataLoader

from .reranker import Reranker


class RerankerTrainer:
    def __init__(self, reranker: Reranker = None, batch_size: int = 64) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.reranker = reranker.to(self.device)

    def train(
        self,
        train_data,
        shuffle: bool = True,
        max_epoch: int = 10,
        loss_func_type: str = None,
        optimizer_type: str = None,
        learning_rate: float = 0.001,
    ):
        self.train_data = train_data
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=2
        )

        size = len(self.train_data)

        self.reranker.train()

        self.loss_func_type = loss_func_type
        loss_func = self.select_loss_func(self.loss_func_type)

        # initialize optimizer
        self.optimizer_type = optimizer_type
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.reranker.parameters(), lr=learning_rate)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.reranker.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.reranker.parameters(), lr=learning_rate)

        train_loss_history = []
        for epoch in range(max_epoch):
            print(f"Epoch {epoch+1}\n-------------------------------")

            batch_loss = []
            for index_batch, sample_batch in enumerate(train_dataloader):
                positive_passage = sample_batch.positive_passage
                negative_passage = sample_batch.negative_passage

                labels = [1] * len(positive_passage.input_ids) + [0] * len(
                    negative_passage.input_ids
                )
                labels[: len(positive_passage)] = 1

                logit = self.reranker(
                    input_ids=torch.cat(
                        [positive_passage.input_ids, negative_passage.input_ids], dim=0
                    ).to(self.device),
                    token_type_ids=torch.cat(
                        [positive_passage.segments, negative_passage.segments], dim=0
                    ).to(self.device),
                    attention_mask=torch.cat(
                        [positive_passage.attn_mask, negative_passage.attn_mask], dim=0
                    ).to(self.device),
                    return_dict=True,
                )

                loss = loss_func(logit.to(self.device), labels.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if index_batch % 10 == 0:
                    current = (index_batch + 1) * self.batch_size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                elif len(positive_passage.input_ids) < self.batch_size:
                    current = size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))

            train_loss_history.append(batch_loss)
            print()

        self.train_loss_history = train_loss_history
        print("Reranker Training Complete!")
        return train_loss_history

    def select_loss_func(self, loss_func_type: str):
        if loss_func_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()
