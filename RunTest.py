import glob
# from image import *
from model import MDBNet as CSRNet
from torch.autograd import Variable
import torch
import os
import math
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2
import visdom


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

root = '/home/joyies/Crowd_Prediction/data/Shanghai/'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
T = './q/'
test_517 = 'qqvideo/'
path_sets = [test_517]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet()
model = model.cuda()
model.eval()

checkpoint = torch.load('B_lr_5model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0
mse = 0
time_sum = 0
# vis = visdom.Visdom()
for i in range(len(img_paths)):
    start = time.time()
    print(img_paths[i])
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    _, h, w = img.size()
    # gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    # groundtruth = np.asarray(gt_file['density'])
    img = img.unsqueeze(0)
    img = torch.nn.functional.upsample(img, [int(h/2), int(w/2)], mode='bilinear')
    output = model(img)
    # t = output.detach().cpu().sum().numpy()-np.sum(groundtruth)
    # mae += abs(t)
    # mse += t*t
    end = time.time()
    time_sum += (end-start)
    # # print("start:", start, "  end:", end)
    # print(output.detach().cpu().sum().numpy())
    file = open('result.txt', 'a+')
    file.write(str(['NO: ', img_paths[i].split('/')[-1],' number: ', output.detach().cpu().sum().numpy()]))
    file.write('\n')
    file.close()

    im = output.squeeze(0)
    im = im.squeeze(0)
    im = im.detach().cpu().numpy()
    image_path = img_paths[i].split('\\')[-1]
    plt.axis('off')
    plt.imshow(im, interpolation='bilinear', cmap=cm.jet)
    plt.savefig('qqvideo-image/'+image_path)

    # heatmap = plt

# print("time_sum:", time_sum, "  avg_time:", time_sum/len(img_paths))



