import torch
from classifier import GarbageClassifier
import time
from torchvision import transforms
from PIL import Image

class_id2name = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回收物', 3: '有害垃圾'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
num_classes = len(class_id2name)
model_name = 'resnext101_32x16d'
model_path = './models/checkpoint/best'
GCNet = GarbageClassifier(num_classes)
GCNet.model.to(device)  # 设置模型运行环境
state_dict = torch.load(model_path)['state_dict']  # state_dict=torch.load(model_path)

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
# load params
GCNet.model.load_state_dict(new_state_dict)
GCNet.model.eval()

# (1)载入图片
img_path = 'data/garbage_classify_4/val/1/img_2206.jpg'  # 图片路径
img = Image.open(img_path)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放最大边=256
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),  # 归一化[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
# (2)特征提取
input_img = preprocess(img)  # c,h,w,=3,224,244
feature = input_img.unsqueeze(0)  # b,c,h,w =1,3,224,224
feature = feature.to(device)

# 传入特征进行预测
with torch.no_grad():
    outputs = GCNet.model.forward(feature)  # ?

# 通过softmax 获取每个label的概率
outputs = torch.nn.functional.softmax(outputs[0], dim=0)
pred_list = outputs.cpu().numpy().tolist()
label_c_mapping = {}
for i, prob in enumerate(pred_list):
    label_c_mapping[int(i)] = prob

# 按照prob 降序，获取topK = 4
dict_list = []
for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:1]:
    label = int(label_prob[0])
    result = {'label': label, 'c': label_prob[1], 'name': class_id2name[label]}
    dict_list.append(result)

print(dict_list)
