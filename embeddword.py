# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)

hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)




# mylist = [u'上海市',u'南京',u'南京']
# myset = set(mylist)
#
# listo=list()
#
# for each in myset:
#     each=unicode(each, encoding="utf-8")
    # listo.append(each)
    # print(each)
#
# print(type(listo))



# print(listo)



mylist = ['上海市','南京','南京']
myset = set(mylist)

listo=list()

for each in myset:
    print(type(each))
    each=unicode(each, encoding="utf-8")
    print(type(each))
    listo.append(each)
    print(each)

print(type(listo))



print(listo)



word_to_ix = {'上海市': 0, '南京': 1}
embeds = nn.Embedding(2, 5)

hello_idx = torch.LongTensor([word_to_ix['上海市']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)


word_to_ix = {'上海市': 0, '南京': 1}
embeds = nn.Embedding(2, 9)

hello_idx = torch.LongTensor([word_to_ix['上海市']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)



word_to_ix = {'上海': 0, '南京': 1}
embeds = nn.Embedding(2, 5)

hello_idx = torch.LongTensor([word_to_ix['上海']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)



# try1=['上海']
# unicode(try1, encoding="utf-8")
# try1=['上海','云南']
# unicode(try1, encoding="utf-8")
