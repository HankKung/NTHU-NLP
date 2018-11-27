
# coding: utf-8

# In[14]:


# pip install wikipedia
# nltk.download("wordnet")
import os
import wikipedia
import nltk
from collections import defaultdict
from nltk.corpus import wordnet as wn
# downlaod wordnet 3.0
#nltk.download("wordnet")
from gensim.models import KeyedVectors
import jieba
import csv


# In[16]:


model_path = '/Users/7ckung/Desktop/Lab2/udn.word2vec.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
word_in_model = model.wv


# # 9 words 54 synsets to find wiki pages, use pos="n" only.

# In[25]:


# Load synsets
target_synsets = []
output = []
i=0
for line in open("/Users/7ckung/Desktop/Lab2/lab02_input.txt"):
    output.append([])
    word, synset, offset = line.strip().split("\t")
    output[i].append(word)
    output[i].append(synset)
    i+=1
    target_synsets.append((word, wn.synset_from_pos_and_offset('n', int(offset))))
            
print("The amount of target synsets :%d" %(len(target_synsets)))


# In[18]:


wn_ec ={}
for line in open("/Users/7ckung/Desktop/Lab2/wn.ec.noun.txt"):
    offset = line.strip().split("\t")[1]
    ch = line.strip().split("\t")[2]
    if offset not in wn_ec:
        wn_ec[offset] = ch


# In[19]:


ec_link ={}
id_c ={}
for line in open("/Users/7ckung/Desktop/Lab2/ec.link.txt"):
    page_id = en = line.strip().split("\t")[0]
    en = line.strip().split("\t")[1]
    ch = line.strip().split("\t")[2].replace('->', '')
    ec_link[str(ch)] = str(en)
    id_c[str(ch)] = int(page_id)


# In[26]:


c=0
lenth=0
for sense in target_synsets:
    offset = sense[1].offset()
    ch = wn_ec[str(offset)]
    ch_l = []
    if '|' in ch:
        ch_l = ch.split('|')
    else:
        ch_l.append(ch)
    find = False
    in_dict = False
    for ch_item in ch_l:
        if ch_item in ec_link:
            in_dict = True
            temp = str(ch_item).replace('_','')
            temp = temp.replace('(', '')
            temp = temp.replace(')', '')
            try:
                page = wikipedia.page(ec_link[temp])
            except:
                find = False

            else:
                print(page.url)
                output[c].append(page.url)
                c+=1
                find = True
                break
    
    if in_dict and find:
        continue
    
    if in_dict and not find:
        for ch_item in ch_l:
            if ch_item in ec_link:
                in_dict = True
                temp = str(ch_item).replace('_','')
                temp = temp.replace('(', '')
                temp = temp.replace(')', '')
                try:
                    page = wikipedia.page(pageid= id_c[temp])
                except:
                    find = False
                else:
                    print(page.url)
                    output[c].append(page.url)
                    c+=1
                    find = True
                    break
                
                      
    test_b=True
    if not in_dict or not find:
        p=0
        good_ch=''
        temp=''
        for ch_item in ch_l:
            temp = str(ch_item).replace('_','')
            temp = temp.replace('(', '')
            temp = temp.replace(')', '')
            if temp in word_in_model:
                for key in ec_link:
                    if key in word_in_model:
                        temp_p = model.similarity(key, temp)
                        if temp_p>p:
                            p = temp_p
                            good_ch = key
            else:
                temp = list(jieba.cut(temp))[-1]
                if temp in word_in_model:
                    for key in ec_link:
                        if key in word_in_model:
                            temp_p = model.similarity(key, temp)
                            if temp_p>p:
                                p = temp_p
                                good_ch = key
        try:
            page = wikipedia.page(ec_link[good_ch])                 
        except:
            test_b = False
        else:
            print(page.url)
            output[c].append(page.url)
            c+=1
        
        if not test_b:
            try:
                page = wikipedia.page(pageid= id_c[good_ch])
            except:
                print('can not find the link of ', ch_l)
                output[c].append('Can not match to Wiki')
                c+=1

            else:
                print(page.url)
                output[c].append(page.url)
                c+=1
    
c


# In[27]:


with open('106062527_lab02_output.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(54):
        tsv_writer.writerow(output[i])

