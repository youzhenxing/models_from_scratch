from operator import index

import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F


class ToyDataset(Dataset):
    def __init__(self,X,y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x,one_y
    def __len__(self):
        return self.labels.shape[0]

class MultiLayersPerceptionExample(torch.nn.Module):
    def __init__(self,input_dims,output_dims):
        super().__init__()
        self.layer = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(input_dims, 30),
            torch.nn.ReLU(),

            # 2nd
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output
            torch.nn.Linear(20, output_dims),
        )
    def forward(self,x):
        logits = self.layer(x)
        return logits


if __name__ == '__main__':
    torch.manual_seed(123)
    X_train = torch.tensor([
       [-1.2,3.1],
       [-0.9,2.9],
       [-0.5,2.6],
       [2.3,-1.1],
       [2.7,-1.5]
    ])

    y_train = torch.tensor([0,0,0,1,1])

    X_test = torch.tensor([
        [-0.8,2.8],
        [2.6,-1.6],
    ])

    y_test = torch.tensor([0, 1])


    train_ds = ToyDataset(X_train,y_train)
    test_ds = ToyDataset(X_test,y_test)

    print(train_ds.__len__())

    train_loader = DataLoader(dataset=train_ds,batch_size=2,shuffle=True,num_workers=0)
    test_loader = DataLoader(dataset=test_ds,batch_size=2,shuffle=False,num_workers=0)

    for idx,(x,y) in enumerate(train_loader):
        print(f"Batch {idx + 1}:",x,y)

    model = MultiLayersPerceptionExample(input_dims=2,output_dims=2)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.5)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #### logging
            print(f"Epoch:{epoch + 1:03d}/{num_epochs:03d}"f"| Batch{batch_idx + 1:03d}/{len(train_loader):03d}"
                  f"| Train Loss: {loss:.2f}")
    model.eval()

    with torch.no_grad():
        outputs = model(X_train)
        print('outputs:',outputs)
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs,dim=1)
    predictions = torch.argmax(probas,dim=1)
    print(predictions)