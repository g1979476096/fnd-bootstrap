import os, sys

# Replace <PROJECT-DIR> placeholder with your project directory path
PROJECT_DIR = '/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/Interpretability'

# Clone PyTorch VQA project from: https://github.com/Cyanogenoid/pytorch-vqa and add to your filepath
sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-vqa"))

# Clone PyTorch Resnet model from: https://github.com/Cyanogenoid/pytorch-resnet and add to your filepath
# We can also use standard resnet model from torchvision package, however the model from `pytorch-resnet`
# is slightly different from the original resnet model and performs better on this particular VQA task
sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-resnet"))

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

from model import Net, apply_attention, tile_2d_over_nd  # from pytorch-vqa
from utils import get_transform  # from pytorch-vqa

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

saved_state = torch.load('./models/2017-08-04_00.55.19.pth', map_location=device)

# reading vocabulary from saved model
vocab = saved_state['vocab']

# reading word tokens from saved model
token_to_index = vocab['question']
# print(token_to_index.get('pad',0),3)

# reading answers from saved model
answer_to_index = vocab['answer']
# print(answer_to_index,'eee')
# {'yes': 0, 'no': 1, '2': 2, '1': 3,

num_tokens = len(token_to_index) + 1

# reading answer classes from the vocabulary
answer_words = ['unk'] * len(answer_to_index)
for w, idx in answer_to_index.items():
    answer_words[idx] = w

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



# Updating weights from the saved model and removing the old model from the memory. And wrap the model with `ModelInputWrapper`.

# In[ ]:


vqa_resnet = VQA_Resnet_Model(vqa_net.module.text.embedding.num_embeddings)

# wrap the inputs into layers incase we wish to use a layer method
# vqa_resnet = ModelInputWrapper(vqa_resnet)
# print(vqa_resnet)
# `device_ids` contains a list of GPU ids which are used for paralelization supported by `DataParallel`
vqa_resnet = torch.nn.DataParallel(vqa_resnet)

# saved vqa model's parameters
partial_dict = vqa_net.state_dict()

state = vqa_resnet.state_dict()
state.update(partial_dict)
vqa_resnet.load_state_dict(state)

num_fc = vqa_resnet.module.classifier.lin2.in_features
num_class = 1  # 二分类
vqa_resnet.module.classifier.lin2 = torch.nn.Linear(num_fc, num_class)

vqa_resnet.to(device)
vqa_resnet.eval()

# This is original VQA model without resnet. Removing it, since we do not need it
del vqa_net

# print(vqa_resnet.module.classifier)

# print(vqa_resnet,'vqa_res')
#  VQA_Resnet_Model(
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



image_size = 448  # scale image to given size and center
central_fraction = 1.0

transform = get_transform(image_size, central_fraction=central_fraction)


def image_to_features(img):
    img_transformed = transform(img)
    img_batch = img_transformed.unsqueeze(0).to(device)
    return img_batch


sents = ['"Kendall Jenner praises Kim Kardashian while appearing on Harper\'s Bazaar She has most definitely arrived on the couture scene and looks like she\'s there to stay.  On Thursday it was revealed Keeping Up With The Kardashians star Kendall Jenner landed the May cover of Harper\'s Bazaar. This has happened less than a year after first hitting the catwalk, which is quite a feat in the supermodel world.  But inside the magazine - in which she has an impressive 10-page feature - the 19-year-old reality star chose to talk not so much about high-end clothes, but rather about her more famous sibling, 34-year-old Kim Kardashian.  Scroll down for video  She made it!: Kendall Jenner appears on the May cover of Harper\'s Bazaar, her first major fashion magazine shoot since she started modeling full time in 2014  But she didn\'t hold back from commenting on her more famous sister: The 19-year-old catwalker dished on Kim Kardashian, pictured here on Thursday in Armenia with sister Khloe behind her  \'I really admire Kim\'s style,\' the daughter of Kris and Bruce Jenner told the fashion monthly.  \'It’s insane. She really knows how to work her style with her body.\'  Kendall\'s praise comes as a surprise after it was rumoured the two TV icons were thought to be feuding. The teen had mentioned in 2014 that having Kim at her fashion shows was not always a good idea as it was a distraction.  Nice look: The budding ubermodel was photographed by designer Karl Lagerfeld in a 10-page feature  In it for the long haul: The 5\'10"" model told the monthly, \'I started this hoping that it would have longevity. I didn’t come into this thinking it’s going to be a fun thing that I’m going to do on the side\'  Her mentor: Jenner also said that Lagerfeld, who shot her Valentino, Armani Privé, Chanel, gives her \'tips, pointers - he makes me feel good when I’m working with him\'  But Kendall, who is the face of Estee Lauder, had nothing but good things to say about Mrs Kanye West (who is at the moment in Armenia with sister Khloe, 30).  \'I think she could be a major fashion icon,\' added the 5\'10"" catwalker.  \'It’s so fun to be in Paris with her, but I’ll walk out and be like, ""Oh, my God, I love my outfit,"" and then I’ll see Kim and I’ll be like, ""My outfit sucks compared to yours!""\'  Chatting up Mrs Kanye West: Jenner said, \'I really admire Kim\'s style. It’s insane. She really knows how to work her style with her body\'; here she is seen on Sunday with (from left) with Khloe, Kim (holding daughter North, aged one) and Kylie in LA  Praise for the lady: \'I think she could be a major fashion icon,\' added the 5\'10"" catwalker said of Kim, 34; here Kendall is pictured in 2009 at sister Khloe\'s wedding to Lamar Odom in Beverly Hills  She really does approve of her style: Kendall also said of Kim, \'It’s so fun to be in Paris with her, but I’ll walk out and be like, ""Oh, my God, I love my outfit,"" and then I’ll see Kim and I’ll be like, ""My outfit sucks compared to yours!""\'; here she is seen (far right) with (from left) Scott Disick, Khloe (holding Mason), Kourtney, Kim and parents Bruce and Kris Jenner  Jenner also talked about what it is like to work with stunners Cara Delevingne and Gigi Hadid.  \'You have to be around these girls for work, for parties, for events, for shows, especially. It’s good to have a set of girlfriends that you can stand being around,\' the sister of Kylie, 17, said.  \'You have to be around these people all the time!\'  As far as modeling, the beauty said she plays to be like Cindy Crawford and Elle Macpherson, as in be a top model for life.  Pals: The looker said that she is happy to be friends with Gigi Hadid because they are together all the time  Another pretty pal: Kendall with Cara Delevingne at a fashion show; that is Lagerfeld behind them  Pouting and smiling: Kendall with mom Kris Jenner as well as Karlie Kloss and Gigi (and some other friends) in an Instagram shot shared in early March  \'I started this hoping that it would have longevity,\' the teen said.  \'I didn’t come into this thinking it’s going to be a fun thing that I’m going to do on the side. It’s something I want to do my whole life.\'  The mannequin - who models Valentino, Armani Privé, Chanel - also dishes about Karl Lagerfeld, who photographed her Bazaar.  \'I’m really shy when I first meet someone — and to meet him was even worse,\' she said.  \'He would say something to me and I would just nod. It was like meeting a crush or something — I was mute. He probably thought I was the biggest weirdo. And then every time we’d see each other, it would just get better and better. Now we’re totally BFFs.\'  She also said he helped her with her career in more ways than one."', '"Jennifer Aniston showing off “Revenge Body” for Brad Pitt, Justin Theroux The Ultimate Source  For all the best News from around the Web"']
img_path = ['/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/dataset/gossip/Images/gossip_train/2uY2BiD8USYFlEIPLzTMniOu2hFdCIel.jpg', '/home/yanwang_nuist/dev_workspace/prj_python/fnd-bootstrap/dataset/gossip/Images/gossip_train/Igqj79fL0LkEEvdLAyc5F0j6pVvORflm.jpg']

def encode_question(question):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question.lower().split()

    while len(question_arr):
        if len(question_arr) > 197:
            question_arr = question_arr[0:197]
            break
        else:
            question_arr.append('pad')
    print(question_arr,len(question_arr))
    vec = torch.zeros(197, device=device).long()
    for i, token in enumerate(question_arr):
        index = token_to_index.get(token, 0)
        vec[i] = index
    # print(vec, torch.tensor(len(question_arr)))
    return vec, torch.tensor(len(question_arr), device=device)

# encode_question(sents[1])


def train_vqa(image_filename, questions):
    img_f_cat = torch.ones(1, 3, 448, 448).requires_grad_().to(device)
    q_cat = torch.ones(1, 197).to(device)
    q_len_cat = torch.ones(1).to(device)
    for img_f, question in zip(image_filename, questions):
        img = Image.open(img_f).convert('RGB')
        image_features = image_to_features(img).requires_grad_().to(device)
        q, q_len = encode_question(question)
        img_f_cat = torch.cat((img_f_cat, image_features), dim=0)
        q_cat = torch.cat((q_cat, q.unsqueeze(0)), dim=0)

        # inputs = (image_features, q)

        # print(image_features.shape, qs.shape, q_len.unsqueeze(0).shape)
        # torch.Size([1, 3, 448, 448]) torch.Size([1, 23])

        q_len_cat = torch.cat((q_len_cat, q_len.unsqueeze(0)), dim=0)

    # print(img_f_cat[1:,:,:,:], q_cat[1:,:], q_len_cat[1:])
    inputs = (img_f_cat[1:,:,:,:], q_cat[1:,:].long())
    ans = vqa_resnet(*inputs, q_len_cat[1:])
    # print(ans)


# train_vqa(img_path, sents)
