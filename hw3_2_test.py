!gdown 1dNO38qQmd44frUxBB9QfQwhokhLz6PP9 -O hw3_data.zip
!unzip -q ./hw3_data.zip

!pip install pyheatmap

!wget -O vit_swin_large_30.ckpt https://www.dropbox.com/s/pze040c6vkgri97/vit_swin_large_30.ckpt?dl=1
# !gdown 1qC6aLLvARZlGx7TxD1pd5FTBETG5x7UY



!pip -q install timm==0.6.11
!pip -q install ftfy regex tqdm
!pip -q install git+https://github.com/openai/CLIP.git
!pip -q install tokenizers==0.13.1
!pip -q install git+https://github.com/bckim92/language-evaluation.git
!python -c "import language_evaluation; language_evaluation.download('coco')"

import sys
from tqdm.notebook import tqdm


import os
import torch
from torchvision.datasets import CIFAR100
import json
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
import csv
import PIL
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torch.nn as nn
import numpy as np
import timm
import math
from tokenizers import Tokenizer
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
path_to_datafile = 'hw3_data/p2_data/'
tokenizer = Tokenizer.from_file("hw3_data/caption_tokenizer.json")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import language_evaluation

"""# Dataset"""

from torch.nn.utils.rnn import pad_sequence
class hw3_2_dataset:
    def __init__(self, filepath, json_file, transform, specific_path):
        self.filepath = filepath
        self.specific_path = specific_path
        self.file_list = []
        self.labels = []
        if 'val' in specific_path.split('/'):
          self.dataset_type = 1
        else:
          self.dataset_type = 0
        for obj in json_file['images']:
          self.file_list.append(obj['file_name'])
          ref_sentences = []
          for x in json_file['annotations']:
            if x['image_id'] == obj['id'] and len(ref_sentences)<5:
              ref_sentences.append(tokenizer.encode(x['caption']).ids)
          self.labels.append(ref_sentences)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.specific_path, self.file_list[idx])
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        # img.show
        transformed_img = self.transform(img)
        img.close()
        y = [torch.as_tensor(c, dtype=torch.long) for c in self.labels[idx]]
        y = pad_sequence(y, padding_value=0)  # type: Tensor
        # print(y.size())
        pad = torch.zeros((100 - y.size(0), 5), dtype = torch.long)
        y = torch.cat((y, pad), dim = 0)
        # print('after', y.size())
        if self.dataset_type:
          return transformed_img, y, self.file_list[idx]
        else:
          return transformed_img, y

# with open(os.path.join(path_to_datafile, 'train.json')) as f:
#   train_json = json.load(f)
with open(os.path.join(path_to_datafile, 'val.json')) as f:
  val_json = json.load(f)
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
# hw3_2_train = hw3_2_dataset(path_to_datafile, train_json, transforms, 'images/train')
hw3_2_val = hw3_2_dataset(path_to_datafile, val_json, transforms, 'images/val')
BATCH_SIZE = 2
# print(len(hw3_2_train))
# train_loader = DataLoader(hw3_2_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(hw3_2_val, batch_size=4, shuffle=False)

"""# Model

## Positional Encoding
"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

"""## Decoder Layer"""

from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs, need_weights = True, average_attn_weights = False)
        # print(output2)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, output2

"""## Decoder & Transformer"""

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, Tensor

class Decoder(nn.Module):
    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        # return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:
        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        # for layer in self.layers:
        #     tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
        #     attns_all = attns
        # # [layer_num, batch_size, head_num, max_len, encode_size**2]

        return tgt_cptn, attns_all


class Transformer(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 dec_ff_dim: int,
                 dec_n_layers: int,
                 dec_n_heads: int,
                 max_len: int,
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super(Transformer, self).__init__()
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        self.encoder = timm.create_model('swin_large_patch4_window7_224', pretrained = False)
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: Tensor,
                captions: Tensor) -> Tuple[Tensor, Tensor]:
        # encode, decode, predict
        images_encoded = self.encoder(images).unsqueeze(1).permute(1,0,2)  # type: Tensor
        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn).permute(1,0,2)  # type: Tensor

        return predictions.contiguous(), attns.contiguous()

"""## Try"""

# import timm 
# timm.list_models()

# print(timm.create_model('vit_base_patch8_224'))

max_len = 75
cp_num = 18022
# src_img = torch.rand(10, 3, 224, 224)  # B, encode, embed
# captions = torch.randint(0, cp_num, (10, 30), dtype=torch.long)
# m_test = Transformer(cp_num, 1000, 2048, 8, 8, max_len, 0.1, 0)
# valus, attns = m_test(src_img, captions)
# print(valus.size())
# print(attns.size())

"""# Train

## Eval
"""

import os
import json
from collections import defaultdict
from PIL import Image
import clip
import torch
import language_evaluation
import torch.nn as nn


def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])
        
        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']


class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.best = 0.
        self.worst = 1.
        self.best_name = ''
        self.worst_name = ''

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.
        
        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")
            curr_score = self.getCLIPScore(image, pred_caption).item()
            # curr_score = self.getCLIPScore(image, pred_caption)
            if curr_score>self.best:
              self.best = curr_score
              self.best_name = image_path
            if curr_score<self.worst:
              self.worst = curr_score
              self.worst_name = image_path
            total_score += curr_score
        print(self.best_name, ' ', self.best)
        print(self.worst_name, ' ', self.worst)
        print(self.best, self.worst)
        # print(total_score)
        # print(total_score.item())
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        cap = caption.split()
        if len(cap) > 75:
            cap = cap[0:75]
        symbol = " ";
        caption = symbol.join(cap)
        text_inputs = torch.cat([clip.tokenize(caption)]).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            text_embedding = self.model.encode_text(text_inputs)
        w = 2.5
        similarity = torch.cosine_similarity(image_embedding,text_embedding)
        return w * similarity


def eval_score(json_name):
    # Read data
    predictions = readJSON(json_name)
    annotations = readJSON('hw3_data/p2_data/val.json')

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore
    clip_score = CLIPScore()(predictions, 'hw3_data/p2_data/images/val/')
    # clip_score = 0
    
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")

"""## Train"""

# model = Transformer(cp_num, 1000, 2048, 8, 8, max_len, 0.1, 0)
# torch.save(model.state_dict(), f'./vit_swin_large_19.ckpt')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model = Transformer(cp_num, 1000, 2048, 8, 8, max_len, 0.1, 0)
model.load_state_dict(torch.load(f'./vit_swin_large_30.ckpt', map_location = device))
model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

from pyheatmap.heatmap import HeatMap

from tqdm.notebook import tqdm
from torch import Tensor
import clip
from torch.autograd import Variable

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# EPOCH = 50
evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])
model.eval()

dictionary = dict()
# loss_all = 0
# p3_imgs = ['bike.jpg', 'girl.jpg', 'sheep.jpg', 'ski.jpg', 'umbrella.jpg']
# hw3_data/p2_data/images/val/000000392315.jpg
# hw3_data/p2_data/images/val/000000141426.jpg
p3_imgs = ['sheep.jpg']
# progress2 = tqdm(total = len(p3_imgs))
imgs = None
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
for batch_idx, img_name in enumerate(p3_imgs):
      img_path = os.path.join('hw3_data/p3_data/images', img_name)
      img = Image.open(img_path).convert('RGB')
      transformed_img = transforms(img)
      img.close()
      if imgs == None:
        imgs = transformed_img.to(device).unsqueeze(0)
      else:
        imgs = torch.cat((imgs, transformed_img.to(device).unsqueeze(0)), dim = 0)
# print(imgs.size())
imgs: Tensor  # images [1, 3, 256, 256]
cptns_all: Tensor  # all 5 captions [1, lm, cn=5]
lens: Tensor  # lengthes of all captions [1, cn=5]
k = 1
# start: [1, 1]
# imgs = transformed_img.to(device).unsqueeze(0)
start = torch.full(size=(imgs.size(0), 1),
                    fill_value=2,
                    dtype=torch.long,
                    device=device)
    # while start.size(1) <= (max_len - 2) and start.nelement():
imgs.requires_grad = True
n = 2
plt.figure(figsize=(18,12))
plt.subplot(4,6,1)
original = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
               std=[1/0.229, 1/0.224, 1/0.225])(imgs[0])
plt.imshow(original.permute(1,2,0).to('cpu').detach().numpy())
plt.title(f"{tokenizer.decode([2], skip_special_tokens= False)}")
while start.size(1) <= (max_len - 2):
    
    logits, attns = model(imgs, start)
    # print(attns.size())
    logits: Tensor  # [k=1, 1, vsc]
    attns: Tensor  # [ln, k=1, hn, S=1, is]
    # print(logits.size())
    
    next_word_id = torch.max(logits, dim=2)[1][:,-1].unsqueeze(1)
    # print(torch.max(logits, dim=2)[0][:,-1])
    # print(next_word_id)
    out = Variable(torch.zeros(logits.size()).to(device))
    # print(out.size())
    out[:, -1, next_word_id.squeeze()] = 1
    # logits[:, -1, :].backward(out[:, -1, :])
    # print(logits.size())
    # print(attns.size())
    # attns[7][-1].backward(attns[7][-1])
    attn_out = Variable(torch.zeros(attns[7].size()).to(device))
    print(attn_out.size(), attns.size())
    point = torch.max(attns[7][-1], dim=-1)[1][0]
    # print(point)
    attn_out[-1][-1][point.item()] = 1
    attns[7].backward(attn_out)
    # print(attns)
    # print(imgs.grad.max())
    new = imgs.grad
    new = torch.abs(imgs.grad).mean(dim = 1).repeat(1, 3, 1, 1)
    # new = imgs.grad.mean(dim = 1).repeat(1, 3, 1, 1)
    # print(logits.size())
    # print(imgs.grad)
    if next_word_id[i].item():
      for i in range(len(p3_imgs)):
        # print('1', new[i].mean(), new[i].min(), new[i].max())
        # new[i] -= new[i].min()
        # new[i] /= torch.max(new[i])
        # new[i] = torch.log(new[i]+1e-4)
        # print('2', new[i].min())
        new[i] -= new[i].mean()
        new[i] /= torch.std(new[i])*1
        original = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])(imgs[i])
        plt.subplot(4,6,n)
        plt.imshow(original.permute(1,2,0).to('cpu').detach().numpy())
        temp = new[i].permute(1,2,0).to('cpu').detach().tolist()
        # print(type(temp), len(temp), len(temp[0]))
        # hm = HeatMap(temp)
        # hit_img = hm.heatmap(r = 100)
        # print(type(hit_img))
        plt.imshow(temp, alpha = 0.4)
        plt.title(f"{tokenizer.decode([next_word_id[i].item()], skip_special_tokens= False)}")
        # plt.imshow(show_img.to('cpu').detach().numpy())
        # plt.show()
        # print('n: ', tokenizer.decode([next_word_id[i].item()], skip_special_tokens= False))
    n+=1
    next_word_id = torch.as_tensor(next_word_id).to(device)
    imgs.grad.data.zero_()
    jump = True
    for wo in range(next_word_id.size(0)):
        if next_word_id[wo][0] != 0:
            jump = False
            break
    if jump:
        break
    start = torch.cat(
        (start, next_word_id), dim=1)
start = start.cpu().tolist()
plt.tight_layout()
plt.show()
for word in range(len(start)):
    word_string = tokenizer.decode(start[word], skip_special_tokens= True)
    print(word_string)
        # name = os.path.splitext(file_names[word])[0]
        # dictionary[name] = word_string
      # progress2.update(1)

from tqdm.notebook import tqdm
from torch import Tensor
import clip
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# EPOCH = 50
evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])
model.eval()
progress2 = tqdm(total = math.ceil(len(val_loader)))
dictionary = dict()
# loss_all = 0
for batch_idx, (imgs, cptns_all, file_names) in enumerate(val_loader):
      imgs: Tensor  # images [1, 3, 256, 256]
      cptns_all: Tensor  # all 5 captions [1, lm, cn=5]
      lens: Tensor  # lengthes of all captions [1, cn=5]
      k = 1
      # start: [1, 1]
      imgs = imgs.to(device)
      start = torch.full(size=(imgs.size(0), 1),
                          fill_value=2,
                          dtype=torch.long,
                          device=device)
      with torch.no_grad():
          # while start.size(1) <= (max_len - 2) and start.nelement():
          while start.size(1) <= (max_len - 2):
              logits, attns = model(imgs, start)
              logits: Tensor  # [k=1, 1, vsc]
              attns: Tensor  # [ln, k=1, hn, S=1, is]
              # print(logits.size())
              
              next_word_id = torch.max(logits, dim=2)[1][:,-1].unsqueeze(1)
              # print('n: ', next_word_id)
              next_word_id = torch.as_tensor(next_word_id).to(device)
              jump = True
              for wo in range(next_word_id.size(0)):
                  if next_word_id[wo][0] != 0:
                      jump = False
                      break
              if jump:
                  break
              start = torch.cat(
                  (start, next_word_id), dim=1)
          start = start.cpu().tolist()
          
          for word in range(len(start)):
              word_string = tokenizer.decode(start[word], skip_special_tokens= True)
              name = os.path.splitext(file_names[word])[0]
              dictionary[name] = word_string
      progress2.update(1)
with open(f'sample_swin_large_19.json', "w") as outfile:
    json.dump(dictionary, outfile)
eval_score(f'sample_swin_large_19.json')

eval_score(f'sample_swin_large_19.json')

from google.colab import files
files.download('sample_swin_large_19.json')