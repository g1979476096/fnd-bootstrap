# print(1)

import torch
import torch.nn as nn

a = torch.tensor([[1.],[0.],[1.],[0.],[1.]]).cuda()
print(a,a.shape)

b = torch.ones_like(a)

# print(b)

c = torch.where(a>0.5,0.4,1.6)
print(c)
# dim0, dim1 = a.shape
# for i in range(dim0):
#     for j in range(dim1):
#         # b[i][j] = a[i][j]
#         if a[i][j] == 1:
#             b[i][j] = 0.4
#         else:
#             b[i][j] = 1.6
# print(b)

# criterion = nn.BCELoss(weight=weights)
criterion = nn.BCEWithLogitsLoss(weight=b.cuda()).cuda()

target = torch.tensor([[12.],[1,],[-22.],[222],[-3]]).cuda()
loss = criterion(target,a)
print(loss)
# tensor(49.6724)
# tensor(73.4641)





# # 假设我们有一个二分类问题
# num_classes = 2
#
# # 假设我们有一个训练集，其中每个样本都有一个标签（0表示负例，1表示正例）
# #
# target = torch.randint(0, num_classes, (100,),dtype=torch.float32)
# print(target)
# #
# labels = torch.randint(0, num_classes, (100,),dtype=torch.float32)
# print(labels)
#
# # 计算正负例的数量
# pos_count = (labels == 1).sum().item()
# neg_count = (labels == 0).sum().item()
# print(pos_count,neg_count)
#
# # 计算类别平衡的倒数
# total_count = pos_count + neg_count
# pos_weight = total_count / (2.0 * pos_count)
# neg_weight = total_count / (2.0 * neg_count)
# print(pos_weight,neg_weight)
#
# # 创建权重张量
# weights = torch.tensor([neg_weight, pos_weight])
# print(weights)
#
# # 定义BCELoss，并传入权重
# weights = torch.randint(0, num_classes, (100,),dtype=torch.float32)
#
# criterion = nn.BCELoss(weight=weights)
# # criterion = nn.BCELoss()
#
# loss = criterion(target,labels)
# print(loss)
# # tensor([1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
# #         1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
# #         1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0,
# #         0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
# #         0, 1, 0, 1])
# # 54 46
# # 0.9259259259259259 1.0869565217391304
# # tensor([1.0870, 0.9259])



# # 假设我们有一个三类分类问题
# num_classes = 3
#
# # 假设我们有一个训练集，其中每个样本都有一个标签（从0到num_classes-1）
# labels = torch.randint(0, num_classes, (100,))
# print(labels)
# # 计算每个类别的频率
# class_counts = torch.bincount(labels)
# class_weights = 1.0 / class_counts
# print(class_counts)
# print(class_weights)
#
# # 创建权重张量
# weights = torch.zeros(num_classes)
# print(weights)
# weights[:len(class_weights)] = class_weights
# print(weights)
#
# # 定义CrossEntropyLoss，并传入权重
# criterion = nn.CrossEntropyLoss(weight=weights)

# tensor([1, 2, 0, 1, 1, 1, 0, 2, 2, 0, 1, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 1, 2,
#         1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1,
#         0, 0, 1, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 1, 2, 2, 0, 0, 2, 0, 0, 2,
#         1, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2,
#         1, 1, 1, 2])
# tensor([30, 32, 38])
# tensor([0.0333, 0.0312, 0.0263])
# tensor([0., 0., 0.])
# tensor([0.0333, 0.0312, 0.0263])
