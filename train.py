import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from classifier import GarbageClassifier


def save_checkpoint(state, isBest, checkpoint='checkpoint', filename='checkpoint'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # 模型保存
    if isBest:
        model_name = 'best'
        model_path = os.path.join(checkpoint, model_name)
        torch.save(state, model_path)


if __name__ == '__main__':
    args = {
        'data_path': 'data/garbage_classify_4',
        'save_path': 'models/checkpoint',
        'num_classes': 4,
        'batch_size': 32,
        'num_workers': 8,
        'lr': 0.001,
        'epochs': 2,
    }

    # (1) 数据载入
    data_path = args['data_path']
    train_path = data_path + '/train'
    val_path = data_path + '/val'
    save_path = args['save_path']
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 缩放最大边=256
        transforms.CenterCrop((224, 224)),  # CNN常将图片裁剪为224*224 -> 7*2^5
        transforms.ToTensor(),  # 归一化[0,1]
        # 使用了使用ImageNet的均值和标准差，是根据数百万张图像计算得出的。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    train_data = datasets.ImageFolder(root=train_path, transform=preprocess)
    val_data = datasets.ImageFolder(root=val_path, transform=preprocess)

    train_loader = DataLoader(
        train_data,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=True)

    val_loader = DataLoader(
        val_data,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=False)

    # (2) 模型初始化
    Classifier = GarbageClassifier(args['num_classes'])

    # (3) 选择损失函数
    criterion = nn.CrossEntropyLoss()

    # (4) 优化器 → 更新参数
    optimizer = torch.optim.Adam(Classifier.model.parameters(), args['lr'])

    # (5) 模型训练及保存
    best_acc = 0
    for epoch in range(1, args['epochs'] + 1):
        print('Epoch : %d / %d' % (epoch, args['epochs']))
        train_loss, train_acc = Classifier.train_model(train_loader, criterion, optimizer)
        test_loss, test_acc = Classifier.test_model(val_loader, criterion, test=None)
        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f' % (train_loss, test_loss, train_acc, test_acc,))
        isBest = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        name = 'checkpoint_' + str(epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': Classifier.model.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, isBest, checkpoint=save_path, filename=name)
        print('Best acc:' + str(best_acc))
