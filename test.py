import torch
import timm
import numpy as np
import torchvision
from PIL import Image
# vit_large_patch16
import cv2
# model_resnet = timm.list_models('*resnet*')
# model = timm.create_model('vit_large_patch16')
# # print(len(model_resnet), model_resnet[:3])
# print(model)
import models_mae
import timm.models.layers.helpers

# with torch.no_grad():
    # path = r'./target.jpg'
    # img_GT = cv2.imread(path, cv2.IMREAD_COLOR)
    #
    # img_GT = img_GT.astype(np.float32) / 255.
    # if img_GT.ndim == 2:
    #     img_GT = np.expand_dims(img_GT, axis=2)
    # # some images have 4 channels
    # if img_GT.shape[2] > 3:
    #     img_GT = img_GT[:, :, :3]
    #
    # ###### directly resize instead of crop
    # img_GT = cv2.resize(np.copy(img_GT), (224, 224),
    #                     interpolation=cv2.INTER_LINEAR)
    #
    # orig_height, orig_width, _ = img_GT.shape
    # H, W, _ = img_GT.shape
    #
    # # BGR to RGB, HWC to CHW, numpy to tensor
    # if img_GT.shape[2] == 3:
    #     img_GT = img_GT[:, :, [2, 1, 0]]
    #
    # img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
    # img_GT = img_GT.unsqueeze(0).cuda()
    # model = models_mae.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
    # model.cuda()
    # # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # model.eval()
    # checkpoint = torch.load('./mae_pretrain_vit_large.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'],strict=False)
    # x = model.forward_ying(img_GT)
    # print(x.shape) # torch.Size([1, 197, 1024])
    # y = torch.mean(x, 1)
    # print(y.shape)


# lis = [i for i in range(-20,20)]
# print(len(lis))

# a = torch.randn(3, 4)
# print(a)
# b = torch.transpose(a,0,1)
# print(b)
#
# mi_loss = -1
#
# b = mi_loss if mi_loss > 0 else 0
# print(b)
# print(torch.tensor(0).item())



import torch
import torch.nn as nn
import torch.nn.functional as F

# nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1))

# a = torch.tensor(3.5)
# print(a)
# print(F.log_softmax(a))
# print(F.logsigmoid(a))
# print(F.sigmoid(a))
# b = torch.randn(3, 4)
# print(F.log_softmax(b, dim=1))

# a = torch.ones(1, 3, 4)
# print(a)
# b = torch.cat((a,a),dim=0)
# print(b, b.shape)
# print(b[1:,:,:])


import torch.nn.functional as F
import numpy as np
a = torch.tensor([[-1199.4813],
        [  653.9160],
        [  271.2624],
        [  -60.7760],
        [ 4341.8574],
        [  424.0250],
        [ -198.9551],
        [  -92.5749]])
# print(a.numpy())
# arr = []
# # print(torch.sigmoid(torch.tensor(2.4)))
# for i in a.numpy():
#     print(i)
#     ar1 = [1, 1]
#     print(i[0])
#     scorr = torch.sigmoid(torch.tensor(i[0]))
#     print(scorr)
#     ar1[0] = 1.0 -scorr.item()
#     ar1[1] = scorr.item()
#     arr.append(ar1)
#
# print(arr)
#
# arr2 = np.array(arr)
# t = torch.from_numpy(arr2)
# print(t)
#

def set_softlabel(output):
    output = output.numpy()
    arr = []
    for i in output:
        ar1 = [1, 1]
        score = torch.sigmoid(torch.tensor(i[0]))
        ar1[0] = 1.0 - score.item()
        ar1[1] = score.item()
        arr.append(ar1)

    arr2 = np.array(arr)
    return torch.from_numpy(arr2)
print(set_softlabel(a))






 # nohup python -u ./interpret_test.py  > interpret_718.log 2>&1 &
