# utf-8=coding
import torch
from model import MDBNet
from torchvision import transforms

import cv2
import visdom
import numpy as np
import time

transform=transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                   ])

# Load net and checkpoint
model = MDBNet()
model = model.cuda()
checkpoint = torch.load('A_lr_5model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# print(cap.isOpened())
vis = visdom.Visdom()
# 480 640
# 1080 1920
# cap = cv2.VideoCapture(0)
# print(cap.isOpened())
# image = np.ones((3, 1080, 1920))
x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# count = 0
renshu = 0
while(1):
    image = np.ones((3, 1080, 1920))
    start = time.time()
    cap = cv2.VideoCapture("rtsp://admin:admin@169.254.78.183:554/stream1")
    # cap = cv2.VideoCapture(0)
    # print(cap.isOpened())
    ret, frame = cap.read()
    time.sleep(2)
    h, w,_ = frame.shape
    image[0, :, :] = frame[:, :, 2]
    image[1, :, :] = frame[:, :, 1]
    image[2, :, :] = frame[:, :, 0]
    image_torch = torch.tensor(image, dtype = torch.float)
    image_torch = transform(image_torch)
    image_torch = image_torch.unsqueeze(0).cuda()
    # print(image_torch.size())
    # image_torch = torch.nn.functional.interpolate(image_torch, [int(h/2), int(w/2)])
    output = model(image_torch)
    print(output.detach().cpu().squeeze(0).squeeze(0).numpy().shape)
    # count = output.detach().cpu().sum().numpy()

    output = output.detach().cpu().squeeze(0).squeeze(0).numpy()
    output = np.flip(output, 0)
    # output = np.flip(output, 1)
    count = output.sum()
    if count < 0:
        count = -count
    # print(time.time()-start)

    # x[0:8] = x[1:9]
    # x[9] = count
    vis.image(image, win='image_history')
    vis.heatmap(output,
                  win='heatmap',
                  opts=dict(
                      colormap='Jet',
                      # title='Count: ' + str(np.around(count)),
                      bar=None,
                      xmax=.5
                  )
                  )
    # if renshu < 9:
    #   # x[9] = count
    #   vis.image(image, win='image_history')
    #   vis.heatmap(output,
    #               win='heatmap',
    #               opts=dict(
    #                   colormap='Jet',
    #                   title='Count: ' + str(np.around(count)),
    #                   bar=None,
    #                   xmax=.3
    #               )
    #               )
    # else:
    #     # x[9] = (x[0:8]+count)/10.0
    #     vis.image(image, win='image_history')
    #     vis.heatmap(output,
    #                 win='heatmap',
    #                 opts=dict(
    #                     colormap='Jet',
    #                     title = 'Count: ' + str(np.around(np.mean(x))+10),
    #                     bar = None,
    #                     xmax = .1
    #                 )
    #                 )
    print(output.sum())
    renshu += 1

cap.release()
cv2.destroyAllWindows()
