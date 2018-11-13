import torch

class BatchWrapper:
    def __init__(self, dl, x, y):
        self.dl, self.x, self.y = dl, x, y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x)

            if self.y is not None:
                y = getattr(batch, self.y).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)
