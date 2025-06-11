#!/usr/bin/env python
# coding: utf-8

# # Model interpretation for Visual Question Answering
# 

# In this notebook we demonstrate how to apply model interpretability algorithms from captum library on VQA models. More specifically we explain model predictions by applying integrated gradients on a small sample of image-question pairs. More details about Integrated gradients can be found in the original paper: https://arxiv.org/pdf/1703.01365.pdf
# 
# As a reference VQA model we use the following open source implementation:
# https://github.com/Cyanogenoid/pytorch-vqa
#   
#   **Note:** Before running this tutorial, please install the `torchvision`, `PIL`, and `matplotlib` packages.

# In[ ]:


import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Replace <PROJECT-DIR> placeholder with your project directory path
PROJECT_DIR = '/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/Interpretability'

# Clone PyTorch VQA project from: https://github.com/Cyanogenoid/pytorch-vqa and add to your filepath
sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-vqa"))

# Clone PyTorch Resnet model from: https://github.com/Cyanogenoid/pytorch-resnet and add to your filepath
# We can also use standard resnet model from torchvision package, however the model from `pytorch-resnet` 
# is slightly different from the original resnet model and performs better on this particular VQA task
sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-resnet"))


# In[ ]:


import threading
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import resnet  # from pytorch-resnet

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from model import Net, apply_attention, tile_2d_over_nd # from pytorch-vqa
from utils import get_transform # from pytorch-vqa

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Loading VQA model

# VQA model can be downloaded from: 
# https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth

# In[ ]:

# dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


GT_size = 224
word_token_length = 197  # identical to size of MAE
image_token_length = 197
token_chinese = BertTokenizer.from_pretrained('bert-base-chinese')
token_uncased = BertTokenizer.from_pretrained('bert-base-uncased')

ref_token_id = token_chinese.pad_token_id


def collate_fn_chinese(data):
    """ In Weibo dataset
        if not self.with_ambiguity:
            return (content, img_GT, label, 0), (GT_path)
        else:
            return (content, img_GT, label, 0), (GT_path), (content_ambiguity, img_ambiguity, label_ambiguity)
    """
    # print(data, 'data_chinese', len(data))
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    # print(sents, image, image_aug, labels, category, GT_path)
    token_data = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=word_token_length,
                                   return_tensors='pt',
                                   return_length=True)
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = token_data['input_ids']
    # print(input_ids, input_ids.shape)  torch.Size([16, 197])
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    # print(image.shape, 'imshape') torch.Size([16, 3, 224, 224])  将len=16的list组合成tensor
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)
    # print(labels.shape, category.shape)
    #torch.Size([16]) torch.Size([16])  batch size

    # yan: len(item) <= 2则没有with_ambiguity
    if len(item) <= 2:
        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path
    else:
        sents1 = [i[2][0] for i in data]
        image1 = [i[2][1] for i in data]
        labels1 = [i[2][2] for i in data]
        token_data1 = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      max_length=word_token_length,
                                                      return_tensors='pt',
                                                      return_length=True)

        input_ids1 = token_data1['input_ids']
        attention_mask1 = token_data1['attention_mask']
        token_type_ids1 = token_data1['token_type_ids']
        image1 = torch.stack(image1)
        labels1 = torch.LongTensor(labels1)

        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path, \
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)

def collate_fn_english(data):
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    token_data = token_uncased.batch_encode_plus(batch_text_or_text_pairs=sents,
                                                 truncation=True,
                                                 padding='max_length',
                                                 max_length=word_token_length,
                                                 return_tensors='pt',
                                                 return_length=True)

    # yan:input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = token_data['input_ids']
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    if len(item) <= 2:
        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path
    else:
        sents1 = [i[2][0] for i in data]
        image1 = [i[2][1] for i in data]
        labels1 = [i[2][2] for i in data]
        token_data1 = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      max_length=word_token_length,
                                                      return_tensors='pt',
                                                      return_length=True)

        input_ids1 = token_data1['input_ids']
        attention_mask1 = token_data1['attention_mask']
        token_type_ids1 = token_data1['token_type_ids']
        image1 = torch.stack(image1)
        labels1 = torch.LongTensor(labels1)

        return (input_ids, attention_mask, token_type_ids), (image, image_aug, labels, category, sents), GT_path, \
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)


# yan:为了使每次产生的随机数都一样，便于复现  seed()可以看做堆，seed(5) 表示第5堆的数值
seed = 25
torch.manual_seed(seed)
np.random.seed(seed)
import random
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


from data.FakeNet_dataset import FakeNet_dataset
train_dataset = FakeNet_dataset(is_filter=False,
                                        is_train=True,
                                        is_use_unimodal=True,
                                        dataset='gossip',
                                        image_size=GT_size,
                                        data_augment = False,
                                        with_ambiguity=False,
                                        use_soft_label=False,
                                        is_sample_positive=False,
                                        duplicate_fake_times=True,
                                        not_on_12=1,
                                        )
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_english,
                                  num_workers=2, sampler=None, drop_last=True,
                                  pin_memory=True)

thresh = train_dataset.thresh

validate_dataset = FakeNet_dataset(is_filter=False, is_train=False,
                                           dataset='gossip',
                                           is_use_unimodal=True,
                                           image_size=GT_size,
                                           not_on_12=1,
                                           )
validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=False,
                                     collate_fn=collate_fn_english,
                                     num_workers=4, sampler=None, drop_last=False,
                                     pin_memory=True)


#
# from data.weibo_dataset import weibo_dataset
# train_dataset = weibo_dataset(is_train=True, image_size=GT_size,
#                               with_ambiguity=False,
#                               not_on_12=1,
#                               )
# # yan:collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能。
# # collate_fn函数就是手动将抽取出的样本堆叠起来的函数
# # drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
#                           collate_fn=collate_fn_chinese,
#                           num_workers=4, sampler=None, drop_last=True,
#                           pin_memory=True)
# thresh = 0.5
#
# validate_dataset = weibo_dataset(is_train=False, image_size=GT_size, not_on_12=1)
#
# validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=False,
#                                      collate_fn=collate_fn_chinese,
#                                      num_workers=4, sampler=None, drop_last=False,
#                                      pin_memory=True)


items = ''
for i, item in enumerate(train_loader):
    if i == 329:
        items = item
        break



print(items[0][0].shape, items[1][0].shape)


texts, others, GT_path = items
input_ids, attention_mask, token_type_ids = texts
image, image_aug, labels, category, sents = others
input_ids, attention_mask, token_type_ids, image, image_aug, labels, category = \
    to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
    to_var(image), to_var(image_aug), to_var(labels), to_var(category)






# dataset

saved_state = torch.load('./models/2017-08-04_00.55.19.pth', map_location=device)

# reading vocabulary from saved model
vocab = saved_state['vocab']

# reading word tokens from saved model
token_to_index = vocab['question']

# reading answers from saved model
answer_to_index = vocab['answer']
# print(answer_to_index,'eee')
# {'yes': 0, 'no': 1, '2': 2, '1': 3,

num_tokens = len(token_to_index) + 1

# reading answer classes from the vocabulary
answer_words = ['unk'] * len(answer_to_index)
for w, idx in answer_to_index.items():
    answer_words[idx]=w


# Loads predefined VQA model and sets it to eval mode.
# `device_ids` contains a list of GPU ids which are used for paralelization supported by `DataParallel`

# In[ ]:


vqa_net = torch.nn.DataParallel(Net(num_tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval()

# print(vqa_net,'vqa')
# (module): Net(
#     (text): TextProcessor(
#       (embedding): Embedding(15193, 300, padding_idx=0)
#       (drop): Dropout(p=0.5, inplace=False)
#       (tanh): Tanh()
#       (lstm): LSTM(300, 1024)
#     )
#     (attention): Attention(
#       (v_conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (q_lin): Linear(in_features=1024, out_features=512, bias=True)
#       (x_conv): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
#       (drop): Dropout(p=0.5, inplace=False)
#       (relu): ReLU(inplace=True)
#     )
#     (classifier): Classifier(
#       (drop1): Dropout(p=0.5, inplace=False)
#       (lin1): Linear(in_features=5120, out_features=1024, bias=True)
#       (relu): ReLU()
#       (drop2): Dropout(p=0.5, inplace=False)
#       (lin2): Linear(in_features=1024, out_features=3000, bias=True)
#     )
#   )
# ) vqa

# Converting string question into a tensor. `encode_question` function is similar to original implementation of `encode_question` method in pytorch-vqa source code.
# https://github.com/Cyanogenoid/pytorch-vqa/blob/master/data.py#L110
# 
# 

# In[ ]:


def encode_question(question):
    # question_arr = token_chinese.encode(question,max_length=197,pad_to_max_length=True,add_special_tokens=False)
    # # print(question_arr)
    # vec = torch.zeros(197, device=device).long()
    # for i, token in enumerate(question_arr):
    #     if token > 15193:
    #         token = 0
    #     vec[i] = token
    # # print(vec)
    # return vec, torch.tensor(len(question_arr), device=device), token_chinese.tokenize(question)


    """ Turn a question into a vector of indices and a question length """
    question_arr = question.lower().split()
    if len(question_arr) > 197:
        question_arr = question_arr[0:197]
    vec = torch.zeros(len(question_arr), device=device).long()
    for i, token in enumerate(question_arr):
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, torch.tensor(len(question_arr), device=device), question_arr


# # Defining end-to-end VQA model

# Original saved model does not have image network's (resnet's) layers attached to it. We attach it in the below cell using forward-hook. The rest of the model is identical to the original definition of the model: https://github.com/Cyanogenoid/pytorch-vqa/blob/master/model.py#L48

# In[ ]:


class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = resnet.resnet152(pretrained=True)
        self.r_model.eval()
        self.r_model.to(device)

        self.buffer = {}
        lock = threading.Lock()

        # Since we only use the output of the 4th layer from the resnet model and do not
        # need to do forward pass all the way to the final layer we can terminate forward
        # execution in the forward hook of that layer after obtaining the output of it.
        # For that reason, we can define a custom Exception class that will be used for
        # raising early termination error.
        def save_output(module, input, output):
            with lock:
                self.buffer[output.device] = output

        self.r_model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.r_model(x)          
        return self.buffer[x.device]

class VQA_Resnet_Model(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()
    
    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


# In order to explain text features, we must let integrated gradients attribute on the embeddings, not the indices. The reason for this is simply due to Integrated Gradients being a gradient-based attribution method, as we are unable to compute gradients with respect to integers.
# 
# Hence, we have two options:
# 1. "Patch" the model's embedding layer and corresponding inputs. To patch the layer, use the `configure_interpretable_embedding_layer`^ method, which will wrap the associated layer you give it, with an identity function. This identity function accepts an embedding and outputs an embedding. You can patch the inputs, i.e. obtain the embedding for a set of indices, with `model.wrapped_layer.indices_to_embeddings(indices)`.
# 2. Use the equivalent layer attribution algorithm (`LayerIntegratedGradients` in our case) with the utility class `ModelInputWrapper`. The `ModelInputWrapper` will wrap your model and feed all it's inputs to seperate layers; allowing you to use layer attribution methods on inputs. You can access the associated layer for input named `"foo"` via the `ModuleDict`: `wrapped_model.input_maps["foo"]`.
# 
# ^ NOTE: For option (1), after finishing interpretation it is important to call `remove_interpretable_embedding_layer` which removes the Interpretable Embedding Layer that we added for interpretation purposes and sets the original embedding layer back in the model.
# 
# Below I am using the `USE_INTEPRETABLE_EMBEDDING_LAYER` flag to do option (1) if it is True, otherwise (2) if it is False. Generally it is reccomended to do option (2) since this option is much more flexible and easy to use. The reason it is more flexible is it allows your model to do any sort of preprocessing to the indices tensor. It's easier to use since you don't have to touch your inputs.

# In[ ]:


USE_INTEPRETABLE_EMBEDDING_LAYER = False  # set to True for option (1)


# Updating weights from the saved model and removing the old model from the memory. And wrap the model with `ModelInputWrapper`.

# In[ ]:


vqa_resnet = VQA_Resnet_Model(vqa_net.module.text.embedding.num_embeddings)

# wrap the inputs into layers incase we wish to use a layer method
vqa_resnet = ModelInputWrapper(vqa_resnet)

# `device_ids` contains a list of GPU ids which are used for paralelization supported by `DataParallel`
vqa_resnet = torch.nn.DataParallel(vqa_resnet)

num_fc = vqa_resnet.module.module.classifier.lin2.in_features
num_class = 1  # 二分类
vqa_resnet.module.module.classifier.lin2 = torch.nn.Linear(num_fc, num_class)

# vqa_resnet.module.load_state_dict(torch.load('./pkl/weibo/917_e8.pkl')
vqa_resnet.module.load_state_dict(torch.load('./pkl/718_e5.pkl'))
# vqa_resnet.module.load_state_dict(torch.load('./pkl/717_e5.pkl'))
# vqa_resnet.module.load_state_dict(torch.load('./pkl/gossip/1025_e5.pkl'))


# # saved vqa model's parameters
# partial_dict = vqa_net.state_dict()
#
#
# state = vqa_resnet.module.state_dict()
# state.update(partial_dict)
# vqa_resnet.module.load_state_dict(state)


vqa_resnet.to(device)
vqa_resnet.eval()

# This is original VQA model without resnet. Removing it, since we do not need it
del vqa_net

print(vqa_resnet.module.module.classifier)



# print(vqa_resnet,'vqa_res')
# (module): ModelInputWrapper(
#     (module): VQA_Resnet_Model(
#       (text): TextProcessor(
#         (embedding): Embedding(15193, 300, padding_idx=0)
#         (drop): Dropout(p=0.5, inplace=False)
#         (tanh): Tanh()
#         (lstm): LSTM(300, 1024)
#       )
#       (attention): Attention(
#         (v_conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (q_lin): Linear(in_features=1024, out_features=512, bias=True)
#         (x_conv): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
#         (drop): Dropout(p=0.5, inplace=False)
#         (relu): ReLU(inplace=True)
#       )
#       (classifier): Classifier(
#         (drop1): Dropout(p=0.5, inplace=False)
#         (lin1): Linear(in_features=5120, out_features=1024, bias=True)
#         (relu): ReLU()
#         (drop2): Dropout(p=0.5, inplace=False)
#         (lin2): Linear(in_features=1024, out_features=3000, bias=True)
#       )
#       (resnet_layer4): ResNetLayer4(...)
#       (input_maps): ModuleDict(
#       (v): InputIdentity()
#       (q): InputIdentity()
#       (q_len): InputIdentity()
#       )
#   )
# )


# Patch the model's embedding layer if we're doing option (1)

# In[ ]:


if USE_INTEPRETABLE_EMBEDDING_LAYER:
    interpretable_embedding = configure_interpretable_embedding_layer(vqa_resnet, 'module.module.text.embedding')


# Below function will help us to transform and image into a tensor.

# In[ ]:


image_size = 448  # scale image to given size and center
central_fraction = 1.0

transform = get_transform(image_size, central_fraction=central_fraction)
    
def image_to_features(img):
    img_transformed = transform(img)
    img_batch = img_transformed.unsqueeze(0).to(device)
    return img_batch


# Creating reference aka baseline / background for questions. This is specifically necessary for baseline-based model interpretability algorithms. In this case for integrated gradients. More details can be found in the original paper: https://arxiv.org/pdf/1703.01365.pdf

# In[ ]:


PAD_IND = token_to_index['pad']
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
# print(PAD_IND)
# print(token_reference,'=')
# 1636
# <captum.attr._models.base.TokenReferenceBase object at 0x7f9667ea3fa0> =

# In[ ]:


# this is necessary for the backpropagation of RNNs models in eval mode
torch.backends.cudnn.enabled=False


# Creating an instance of layer integrated gradients for option (2); otherwise create an instance of integrated gradients for option (1). Both are equivalent methods to interpret the model's outputs.

# In[ ]:


if USE_INTEPRETABLE_EMBEDDING_LAYER:
    attr = IntegratedGradients(vqa_resnet)
else:
    attr = LayerIntegratedGradients(vqa_resnet, [vqa_resnet.module.input_maps["v"], vqa_resnet.module.module.text.embedding])


# Defining default cmap that will be used for image visualizations 

# In[ ]:


default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)


# Defining a few test images for model intepretation purposes

# In[ ]:


# images = ['./img/vqa/2.jpg',
#           './img/vqa/1.jpg',
#           './img/vqa/elephant.jpg',
#           './img/vqa/siamese.jpg',
#           './img/vqa/zebra.jpg']

images = GT_path

# In[ ]:


def vqa_resnet_interpret(image_filename, questions, targets):
    img = Image.open(image_filename).convert('RGB')
    original_image = transforms.Compose([transforms.Resize(int(image_size / central_fraction)),
                                   transforms.CenterCrop(image_size), transforms.ToTensor()])(img) 
    
    image_features = image_to_features(img).requires_grad_().to(device)
    for question, target in zip(questions, targets):
        q, q_len,quest = encode_question(question)
        
        # generate reference for each sample
        q_reference_indices = token_reference.generate_reference(q_len.item(), device=device).unsqueeze(0)

        inputs = (q.unsqueeze(0), q_len.unsqueeze(0))
        if USE_INTEPRETABLE_EMBEDDING_LAYER:
            q_input_embedding = interpretable_embedding.indices_to_embeddings(q).unsqueeze(0)
            q_reference_baseline = interpretable_embedding.indices_to_embeddings(q_reference_indices).to(device)

            inputs = (image_features, q_input_embedding)
            baselines = (image_features * 0.0, q_reference_baseline)
            
        else:            
            inputs = (image_features, q.unsqueeze(0))
            baselines = (image_features * 0.0, q_reference_indices)

        # print(image_features.shape, q.unsqueeze(0).shape)
        # torch.Size([1, 3, 448, 448]) torch.Size([1, 23])


        ans = vqa_resnet(*inputs, q_len.unsqueeze(0))
        print(ans,'ans')


        # print(ans) tensor([[0.0374]], device='cuda:0', grad_fn=<GatherBackward>)
        # print(ans, ans.shape,'ans')
        #  tensor([[ -9.5934, -11.0648, -15.8490,  ..., -15.5659, -34.7921, -23.7605]],
        #   device='cuda:0', grad_fn=<GatherBackward>)
        #   torch.Size([1, 3000]) ans

        # Make a prediction. The output of this prediction will be visualized later.


        # pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)
        # print(pred)
        # print(answer_idx)
        # tensor([0.5677])
        # tensor([0])
        # target = answer_idx,
        attributions = attr.attribute(inputs=inputs,
                                    baselines=baselines,

                                    additional_forward_args=q_len.unsqueeze(0),
                                    n_steps=30)

        # print(attributions[0].shape, attributions[1].shape,)
        # print(attributions[0], attributions[0].shape)
        # return
        # Visualize text attributions

        text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()

        # print(attributions[1].sum(dim=2).squeeze(0).shape)
        # print(question.split())
        #  pred[0].item(),
        #                                 answer_words[ answer_idx ],
        #                                 answer_words[ answer_idx ],
        #                                 target,

        print(len(attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm), len(quest), 11)
        vis_data_records = [visualization.VisualizationDataRecord(
                                attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
                                15,
                                'real',
                                'real',
                                target,
                                attributions[1].sum(),
                                quest,
                                0.0)]

        aa = visualization.visualize_text(vis_data_records)
        print(aa.data)

        # visualize image attributions
        print(attributions[0].shape,'11')
        original_im_mat = np.transpose(original_image.cpu().detach().numpy(), (1, 2, 0))
        attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        visualization.visualize_image_attr_multiple(attributions_img, original_im_mat,
                                                    ["original_image", "heat_map"], ["all", "absolute_value"],
                                                    titles=["Original Image", "Attribution Magnitude"],
                                                    cmap=default_cmap,
                                                    show_colorbar=True)
        print('Text Contributions: ', attributions[1].sum().item())
        print('Image Contributions: ', attributions[0].sum().item())
        print('Total Contribution: ', attributions[0].sum().item() + attributions[1].sum().item())


# In[ ]:


# the index of image in the test set. Please, change it if you want to play with different test images/samples.
image_idx = 0 # elephant
vqa_resnet_interpret(images[image_idx], [sents[image_idx]], ['real news'])


# 论文上图片
# v load, bsz:8,  item:2 ,image_idx:5
# v load, bsz:8,  item:2 ,image_idx:6
# v load, bsz:8,  item:5 ,image_idx:2
# train load, bsz:8,  item:329 ,image_idx:3

# 其他可以的
# v load, bsz:8,  item:4 ,image_idx:6
# v load, bsz:8,  item:5 ,image_idx:1
# t load, bsz:8,  item:421 ,image_idx:2



# for image_idx in range(0,8):
#   vqa_resnet_interpret(images[image_idx], [sents[image_idx]], ['real news'])

# sentts = """John Stamos Eager To Have A Family: ‘Inspired By George Clooney’s Happiness & Fatherhood’ At 54, John Stamos is finally ready for kids! After announcing his engagement to younger GF Caitlin McHugh, we’ve learned exclusively the star’s excited about possibly becoming a dad. His inspo? George Clooney!  Taking a page from George Clooney‘s, 56, book, John Stamos, 54, is ready to become a father for the first time! The actor popped the question to his girlfriend Caitlin McHugh, 32, earlier this week, and apparently he’s ALREADY got babies on the brain! Despite his age though, John is eager to settle down and raise a fam — after all, if George can make it happen, he can too! But George isn’t the only one who’s inspired John to become a family man. Click here to see adorable pics of celeb dads with their kids.  “John is eager to finally have a family after being inspired by George Clooney’s late-life happiness and fatherhood,” a source close to John shared with HollywoodLife.com EXCLUSIVELY. “At the age of 54, John made the decision to get married again and hopefully start a family. After watching George pull off a similar life change at almost the same age, John feels being a dad could be for him too.” How sweet is THAT? This will be John’s second marriage, just like George, who married Amal Clooney, 39, in 2014. This past June, George and Amal welcomed their very first children: twins Ella and Alexander. We can totally see the same happening for John!  “More than just George however, John has been inspired by so many friends and family who have caring, loving families,” our insider added. “As John has matured, his priorities and perspectives on life have changed too. He is finally ready to settle down, become a father and enter the next phase of his life.” John and Caitlin started dating back in March 2016, and it appears he popped the question in Disneyland! “I asked…she said yes! …And we lived happily ever after💍,” John captioned a sweet Instagram photo that was posted on Oct. 23.  The image is an illustration of a man and woman embracing in front of Cinderella’s castle as fireworks explode in the background. Aw! Hours later, John posted a photograph of himself with Caitlin, both wearing Mickey Mouse ears. “#Forever,” he captioned the image. We cannot wait for this couple’s wedding!  Tell us, HollywoodLifers — would you love to see John become a father? Do you think it’ll happen for him?"""
# images = '/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/dataset/gossip/Images/gossip_test/KHWFHBT7go9EH8acKwboQNAKNCBOPrDq.jpg'
# vqa_resnet_interpret(images, [sentts], ['fake news'])

# In[1]:


# import IPython
# Above cell generates an output similar to this:
# IPython.display.Image(filename='img/vqa/elephant_attribution.jpg')


# In[ ]:


# image_idx = 0 # cat
#
# vqa_resnet_interpret(images[image_idx], [
#     "what is on the picture",
#     "what color are the cat's eyes",
#     "is the animal in the picture a cat or a fox",
#     "what color is the cat",
#     "how many ears does the cat have",
#     "where is the cat"
# ], ['cat', 'blue', 'cat', 'white and brown', '2', 'at the wall'])


# In[2]:


# Above cell generates an output similar to this:
# IPython.display.Image(filename='img/vqa/siamese_attribution.jpg')


# In[ ]:


# image_idx = 2 # zebra

# vqa_resnet_interpret(images[image_idx], [
#     "what is on the picture",
#     "what color are the zebras",
#     "how many zebras are on the picture",
#     "where are the zebras"
# ], ['zebra', 'black and white', '2', 'zoo'])


# In[3]:


# Above cell generates an output similar to this:
# IPython.display.Image(filename='img/vqa/zebra_attribution.jpg')


# As mentioned above, after we are done with interpretation, we have to remove Interpretable Embedding Layer and set the original embeddings layer back to the model.

# In[ ]:


if USE_INTEPRETABLE_EMBEDDING_LAYER:
    remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)

