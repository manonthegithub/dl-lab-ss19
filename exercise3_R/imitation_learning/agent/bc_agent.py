import torch
from agent.networks import CNN

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)


class BCAgent:

    def __init__(self, weights, lr=1e-3, history_length=1):
        # TODO: Define network, loss function, optimizer
        n_classes = 5
        print(DEVICE)
        print(device)
        self.net = CNN(history_length=history_length, n_classes=n_classes).to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        X_batch = torch.tensor(X_batch).to(device)
        y_batch = torch.tensor(y_batch).to(device)
        out = self.net(X_batch)
        loss = self.loss_fn(out, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def predict(self, X):
        with torch.no_grad():
            # TODO: forward pass
            X = torch.tensor(X).to(device)
            self.net.eval()
            outputs = self.net(X)
            outputs = outputs.detach()
            return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
        return file_name
