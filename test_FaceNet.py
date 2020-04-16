from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torchvision import datasets, transforms
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

### Crop the image
img = Image.open('/home/dchen/dataset/CelebA/Img/img_align_celeba/182733.jpg')
mtcnn = MTCNN(image_size=160, margin=0)
img_cropped = mtcnn(img)
print(img_cropped.shape)

### For a model pretrained on VGGFace2 
model = InceptionResnetV1(num_classes=8631).eval()
model.load_state_dict(torch.load('/home/dchen/GM/pretrain_models/20180402-114759-vggface2.pt', map_location=torch.device('cpu')))

### For an untrained 100-class classifier
# model = InceptionResnetV1(classify=True, num_classes=100)
# print(model)

### Calculate embedding (unsqueeze to add batch dimension), torch.Size([1,512])
img_embedding = model(img_cropped.unsqueeze(0))
print(img_embedding.shape)

### Or, if using for VGGFace2 classification, torch.Size([1,8631])
# model.classify = True
# img_probs = model(img_cropped.unsqueeze(0))
# print(img_probs.shape)

'''
if __name__ == "__main__":
    device = torch.device("cuda")
    torch.cuda.set_device(9)

    EPOCH_NUM = 10
    BATCH_SIZE = 32
    LR = 0.001

    transform = transforms.Compose([
        np.float32,
        transforms.CenterCrop(64),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    
    # {'female': 0, 'male': 1}
    trainset = datasets.ImageFolder('/home/dchen/dataset/CelebA/gender/', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, [5, 10])

    
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }
'''