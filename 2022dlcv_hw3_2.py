# -*- coding: utf-8 -*-
"""2022dlcv_hw3_2.ipynb 的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oxrXSOBCSR2M4HIbiE7Thzd_oVm6VpPZ

# Download
"""
# !pip install timm==0.6.11
# !pip install ftfy regex tqdm
# !pip install git+https://github.com/openai/CLIP.git
# !pip install tokenizers==0.13.1
# !pip install git+https://github.com/bckim92/language-evaluation.git
# !python -c "import language_evaluation; language_evaluation.download('coco')"

"""# Preparation"""

import sys
from tqdm import tqdm


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
        #   for obj in json_file['images']:
        #     self.file_list.append(obj['file_name'])
        #     ref_sentences = []
        #     count = 0
        #     for x in json_file['annotations']:
        #       if x['image_id'] == obj['id'] and count<5:
        #         ref_sentences.append(x['caption'])
        #         count += 1
        #     # else:
        #     #   print(obj['id'])
        #     self.labels.append(ref_sentences)
        else:
          self.dataset_type = 0
        for obj in json_file['images']:
          self.file_list.append(obj['file_name'])
          ref_sentences = []
          for x in json_file['annotations']:
            if x['image_id'] == obj['id'] and len(ref_sentences)<5:
              ref_sentences.append(tokenizer.encode(x['caption']).ids)
            # ref_sentences = np.zeros((5, 50))
            # count = 0
            # for x in json_file['annotations']:
            #   if x['image_id'] == obj['id'] and count<5:
            #     for a in range(50):
            #       if a < len(tokenizer.encode(x['caption']).ids):
            #         ref_sentences[count][a] = int(tokenizer.encode(x['caption']).ids[a])
            #       else:
            #         ref_sentences[count][a] = 0
            #     count += 1
            # else:
            #   print(obj['id'])
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

with open(os.path.join(path_to_datafile, 'train.json')) as f:
  train_json = json.load(f)
with open(os.path.join(path_to_datafile, 'val.json')) as f:
  val_json = json.load(f)
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5],
               std=[0.5])
])
hw3_2_train = hw3_2_dataset(path_to_datafile, train_json, transforms, 'images/train')
hw3_2_val = hw3_2_dataset(path_to_datafile, val_json, transforms, 'images/val')
BATCH_SIZE = 8
print(len(hw3_2_train))
train_loader = DataLoader(hw3_2_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(hw3_2_val, batch_size=BATCH_SIZE, shuffle=False)

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
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns

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
        self.encoder = timm.create_model('vit_base_patch8_224', pretrained = False)
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

max_len = 100
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

            total_score += self.getCLIPScore(image, pred_caption)
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
        image_e = self.model.encode_image(image)
        caption_e = self.model.encode_text(caption)
        return 2.5*nn.CosineSimilarity(dim = -1, eps = 1e-8)(image_e, caption_e)


def eval_score(json_name):
    # Read data
    predictions = readJSON(json_name)
    annotations = readJSON("hw3_data/p2_data/val.json")

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore
    # clip_score = CLIPScore()(predictions, "hw3_data/p2_data/images/val/")
    clip_score = 0
    
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")

"""## Train"""

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model = Transformer(cp_num, 1000, 1024, 8, 8, max_len, 0.1, 0)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000003)
model_epoch = 20
model.load_state_dict(torch.load(f'./vit_b4_{model_epoch}.ckpt'))

from tqdm import tqdm
from torch import Tensor
import clip
# clip_model, preprocess = clip.load('ViT-B/32', device)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EPOCH = 20
evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

def clip_gradient(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

# Sizes:
# B:   batch_size
# is:  image encode size^2: image seq len: [default=196]
# vsc: vocab_size: vsz
# lm:  max_len: [default=52]
# cn:  number of captions: [default=5]
# hn:  number of transformer heads: [default=8]
# ln:  number of layers
# k:   Beam Size
print(len(train_loader))
# start

for epoch in range(EPOCH):
    # progress = tqdm(total = math.ceil(len(train_loader)))
    loss_all = 0
    model.train()
    for batch_idx, (imgs, cptns_all) in enumerate(train_loader):
        
        imgs: Tensor  # images [B, 3, 256, 256]
        cptns_all: Tensor  # all 5 captions [B, lm, cn=5]

        # move data to device, and random selected cptns
        imgs = imgs.to(device, dtype=torch.float)

        # imgs = imgs.repeat(5,1,1,1)
        # cptns = cptns_all.squeeze(0).permute(1,0)
        idx = np.random.randint(0, cptns_all.size(-1))
        cptns = cptns_all[:, :, idx].to(device, torch.int32)

        # zero the parameter gradients
        model.zero_grad()

        logits, attns = model(imgs, cptns[:, :-1])
        # print(logits.size())
        logits: Tensor  # [B, lm - 1, vsz]
        attns: Tensor  # [ln, B, hn, lm, is]
        if (not batch_idx%500):
          print('ans: ',tokenizer.decode(cptns[0][:-1].tolist(), skip_special_tokens= True))
          print('logits: ', tokenizer.decode(torch.max(logits, dim=2)[1][0].tolist(), skip_special_tokens= True))
        # print('logits shape:', logits.size())
        # print('logits: ',logits)
        # loss calc, backward
        loss = criterion(logits.reshape(-1, cp_num), cptns[:, 1:].reshape(-1).to(device, dtype = torch.int64))
        loss.backward()
        clip_gradient(optimizer)
        optimizer.step()
        loss_all += loss.item()
        # get predections then alculate some metrics
        # preds = torch.argmax(logits, dim=2).cpu()  # predections
        # targets = cptns_all[:, 1:]  # remove <SOS>
        # progress.update(1)
    torch.save(model.state_dict(), f'./vit_b4_{epoch+model_epoch+1}.ckpt')
    print('Loss = ', loss_all/len(train_loader))
    if epoch>-1:
      model.eval()
    #   progress2 = tqdm(total = math.ceil(len(val_loader)))
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
          # start[:, 0] = 2
          
          # final_word = torch.zeros(start.size(), dtype = torch.long, device = device)
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
              # b = start.size(0)
              # rm = []
              # for check in range(b):
              #   if next_word_id[check][0] == 0 or next_word_id[check][0] == 3:
              #     rm.append(check)
              # padd = torch.zeros((b,1), dtype = torch.long, device = device)
              start = torch.cat(
                  (start, next_word_id), dim=1)
              # final_word = torch.cat(
              #     (final_word, padd), dim=1)
              # final_word[rm, :] = start[rm, :]
              # print(final_word.size())
              # start = start[list(range(b)) not in rm, :].squeeze(0)
              # imgs = imgs[list(range(b)) not in  rm, :].squeeze(0)
              # print(start.size(), imgs.size())
            start = start.cpu().tolist()
            
            for word in range(len(start)):
              word_string = tokenizer.decode(start[word], skip_special_tokens= True)
              name = os.path.splitext(file_names[word])[0]
              dictionary[name] = word_string
        #   progress2.update(1)
      with open(f"sample_{epoch+model_epoch+1}.json", "w") as outfile:
          json.dump(dictionary, outfile)
      print("epoch ", epoch+model_epoch+1)
      eval_score(f"sample_{epoch+model_epoch+1}.json")

