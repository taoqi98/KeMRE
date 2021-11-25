from Knowledge import *
from hypers import *
import json 
import numpy as np
import keras_bert
from gensim.models import Word2Vec



def read_data():
    with open('paper_data/train_data.json') as f:
        TrainData = json.load(f)
    with open('paper_data/test_data.json') as f:
        TestData = json.load(f)
    with open('paper_data/Centorid.json') as f:
        Centorid = json.load(f)
    
    return TrainData,TestData,Centorid


def parse_label_to_numpy(labels,max_entity_num):
    z = np.zeros((max_entity_num,max_entity_num))
    for lid in range(len(labels)):
        label = labels[lid]
        for i in range(len(label)-1):
            for e1 in label[i]:
                for e2 in label[i+1]:
                    z[e1,e2] = 1
    return z

def evaluation(ypred,ytrue,mask):
    pmask = mask.reshape((mask.shape[0],max_entity_num,max_entity_num))
    pmask = pmask>0
    
    ytrue = ytrue.reshape((ytrue.shape[0],max_entity_num,max_entity_num))
    #print(ypred)
    
    acc = ypred==ytrue
        
    acc = (acc*pmask).sum()/pmask.sum()
    
    recall = ypred[ytrue==1].sum()/(ytrue.sum()+0.000001 )
    
    p = ytrue[ypred].sum()/(ypred.sum() + 0.000001)
    
    f1 = 2/(1/p + 1/recall)
    
    return acc,recall, p, f1

def read_bert_word_index(bert_word_index_path):
    with open(bert_word_index_path) as f:
        lines=f.readlines()
    index=0
    bert_word_index={}
    for l in lines:
        l=l.strip()
        bert_word_index[l]=index
        index+=1
    return bert_word_index

def Init_char_embedding(path,voca):
    word2index=Word2Vec.load(path)
    matrix=np.random.normal(size=(len(voca)+1,200))

    for word in voca.keys():
        index=voca[word]
        if word in word2index:
            matrix[index,:]=word2index[word]
    return matrix

def parse_data(Data,Centorid,char_dict,tp_dict,bert_voca):
    
    char_index = len(char_dict) + 1
    tp_index = len(tp_dict) + 1
    
    text = []
    bert_text = []
    entity = []
    entity_type = []
    label = []
    mask = []

    Labels = []
    Knowledge = []
    Masks = []

    ES = []
    ct = 0

    classes = []
    class_dict = {}
    class_index = 0

    for key in Data:
        cls = Data[key]['类别']
        if not cls in class_dict:
            class_dict[cls] = class_index
            class_index += 1
        classes.append(class_dict[cls])

        knowledge = GetKnowledge(Data[key],Centorid[cls])
        label = Data[key]['标注结果']
        knowledge0 = knowledge
        knowledge = parse_knowledge_to_numpy(knowledge,max_entity_num)
        label = parse_label_to_numpy(label,max_entity_num)

        Labels.append(label)
        Knowledge.append(knowledge)

        pmask = np.zeros((max_entity_num,max_entity_num))
        num = len(Data[key]['实体识别结果'])
        pmask[:num,:num] = 1
        Masks.append(pmask)

        ES.append(Data[key]['实体识别结果'])


        g = []
        bg = []
        for char in Data[key]['原文']:
            if not char in char_dict:
                char_dict[char] = char_index
                char_index += 1
            g.append(char_dict[char])

            if char in bert_voca:
                bg.append(bert_voca[char])
            else:
                bg.append(0)


        if len(g)<max_text_length:
            g += [0]*(max_text_length-len(g))
            bg += [0]*((max_text_length-len(bg)))
        else:
            g = g[:max_text_length]
            bg = bg[:max_text_length]
        text.append(g)
        bert_text.append(bg)


        e_texts = []
        e_tps = []
        for e in range(len(Data[key]['实体识别结果'])):
            e = str(e)
            word,tp,sp,ep = Data[key]['实体识别结果'][e]
            g = np.zeros((max_text_length,))
            g[sp:ep] = 1/(ep-sp)
            e_texts.append(g)
            if not tp in tp_dict:
                tp_dict[tp] = tp_index
                tp_index += 1
            e_tps.append(tp_dict[tp])

        e_texts = e_texts + [np.zeros((max_text_length,))]*(max_entity_num-len(e_texts))
        e_tps = e_tps + [0]*(max_entity_num-len(e_tps))
        e_texts = np.array(e_texts)
        e_tps = np.array(e_tps)
        entity.append(e_texts)
        entity_type.append(e_tps)


    Labels = np.array(Labels)
    Knowledge = np.array(Knowledge)
    Masks = np.array(Masks)

    text = np.array(text)
    bert_text = np.array(bert_text)
    bert_seg = np.zeros(bert_text.shape)
    entity = np.array(entity)
    entity_type = np.array(entity_type)

    classes = np.array(classes)
    
    return [text, entity, entity_type, bert_text, bert_seg, Knowledge,], Labels, Masks,


def parse_datav2(Data,char_dict,tp_dict,bert_voca):
    
    char_index = len(char_dict) + 1
    tp_index = len(tp_dict) + 1
    
    text = []
    bert_text = []
    entity = []
    entity_type = []
    label = []
    mask = []

    Labels = []
    Masks = []

    ES = []
    ct = 0

    classes = []
    class_dict = {}
    class_index = 0

    for key in Data:
        cls = Data[key]['类别']
        if not cls in class_dict:
            class_dict[cls] = class_index
            class_index += 1
        classes.append(class_dict[cls])

        label = Data[key]['标注结果']
        label = parse_label_to_numpy(label,max_entity_num)

        Labels.append(label)

        pmask = np.zeros((max_entity_num,max_entity_num))
        num = len(Data[key]['实体识别结果'])
        pmask[:num,:num] = 1
        Masks.append(pmask)

        ES.append(Data[key]['实体识别结果'])


        g = []
        bg = []
        for char in Data[key]['原文']:
            if not char in char_dict:
                char_dict[char] = char_index
                char_index += 1
            g.append(char_dict[char])

            if char in bert_voca:
                bg.append(bert_voca[char])
            else:
                bg.append(0)


        if len(g)<max_text_length:
            g += [0]*(max_text_length-len(g))
            bg += [0]*((max_text_length-len(bg)))
        else:
            g = g[:max_text_length]
            bg = bg[:max_text_length]
        text.append(g)
        bert_text.append(bg)


        e_texts = []
        e_tps = []
        for e in range(len(Data[key]['实体识别结果'])):
            e = str(e)
            word,tp,sp,ep = Data[key]['实体识别结果'][e]
            g = np.zeros((max_text_length,))
            g[sp:ep] = 1/(ep-sp)
            e_texts.append(g)
            if not tp in tp_dict:
                tp_dict[tp] = tp_index
                tp_index += 1
            e_tps.append(tp_dict[tp])

        e_texts = e_texts + [np.zeros((max_text_length,))]*(max_entity_num-len(e_texts))
        e_tps = e_tps + [0]*(max_entity_num-len(e_tps))
        e_texts = np.array(e_texts)
        e_tps = np.array(e_tps)
        entity.append(e_texts)
        entity_type.append(e_tps)


    Labels = np.array(Labels)
    Masks = np.array(Masks)

    text = np.array(text)
    bert_text = np.array(bert_text)
    bert_seg = np.zeros(bert_text.shape)
    entity = np.array(entity)
    entity_type = np.array(entity_type)

    classes = np.array(classes)
    
    return [text, entity, entity_type, bert_text, bert_seg,], Labels, Masks,