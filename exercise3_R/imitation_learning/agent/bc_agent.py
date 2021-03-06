import torch
from agent.networks import CNN


class BCAgent:

    def __init__(self, device, weights= None, lr=1e-3, history_length=1):
        # TODO: Define network, loss function, optimizer
        self.n_classes = 5
        model = CNN(history_length=history_length, n_classes=self.n_classes)
        model.to(device)
        self.net = model
        if weights is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device).float())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        out = self.net(X_batch)
        out = out.view(-1, self.n_classes)
        loss = self.loss_fn(out, y_batch.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, X):
        with torch.no_grad():
            # TODO: forward pass
            self.net.eval()
            outputs = self.net(X)
            return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name, map_location='cpu'))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
        return file_name
