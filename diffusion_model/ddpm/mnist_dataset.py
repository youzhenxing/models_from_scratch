import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor

def download_mnist(img_show = False):
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('mnist data size:', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)
    if img_show == True:
       img.show()
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())

def get_dataloader(batch_size:int):
    #由于DDPM和正态分布关联起来，更希望图像的颜色值范围是[-1,1],这里做一个线性变换
    transform = Compose([ToTensor(), Lambda(lambda x : (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)
    print('mnist data size:', len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_img_shape():
    return (1, 28, 28)


if __name__ == '__main__':
    download_mnist()