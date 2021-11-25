from hypers import *
import keras
from keras import Input
from keras.layers import *
import keras.backend as K
from keras.optimizers import  *
from keras.layers.core import Lambda



def MyLoss(ytrue,ypred):
    logit, mask = ypred

    logit = ytrue*K.log(logit)+(1-ytrue)*K.log(1-logit)
    logit = logit*mask
    logit = K.sum(logit,)
    
    return logit

def get_classifier(dim):
    vec_input = Input(shape=(dim,))
    vec = Dense(256,activation='relu')(vec_input)

    logit = Dense(1,activation='sigmoid')(vec)
    model = keras.Model(vec_input,logit)
    return model

class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None


def infer(model,ipt,mask,theta=0.5):
    pred = model.predict(ipt,verbose=0)
    pred = pred.reshape((pred.shape[0],max_entity_num,max_entity_num,))
    pmask = mask.reshape((mask.shape[0],max_entity_num,max_entity_num))
    pred = pred*(pmask>0)
    pred = pred>theta
    return pred

def get_model(emb_matrix,bert,tp_dict,use_knowledge,use_bert,char_cnn,char_lstm,entity_type,entity_cnn,entity_lstm,dim_knowledge):
    embedding_matrix = emb_matrix
    texts_input = Input(shape=(max_text_length,),dtype='int32')
    entity_input = Input(shape=(max_entity_num,max_text_length),dtype='float32')
    entity_type_input = Input(shape=(max_entity_num,),dtype='int32')
    bert_text_input = Input(shape=(max_text_length,),dtype='int32')
    bert_seg_input = Input(shape=(max_text_length,),dtype='int32')
    knowledge_input = Input(shape=(max_entity_num,max_entity_num),dtype='int32')

    embedding_layer = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)
    text_embedding = embedding_layer(texts_input)
    bert_embedding = bert([bert_text_input,bert_seg_input])
    bert_embedding = RemoveMask()(bert_embedding)
    
    if use_bert:
        text_embedding = keras.layers.Concatenate(axis=-1)([text_embedding,bert_embedding])
        text_embedding = Dropout(0.2)(text_embedding)
        text_embedding = Dense(400)(text_embedding)
        
    
    text_vecs = text_embedding
    if char_cnn:
        text_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(text_vecs)
    if char_lstm:
        text_vecs = Bidirectional(LSTM(200,return_sequences=True))(text_vecs) #(max_text_length,400)
    
    entity_type_embedding_layer = Embedding(len(tp_dict),200,trainable=True)
    entity_type_emb = entity_type_embedding_layer(entity_type_input)
    entity_type_emb = Dropout(0.2)(entity_type_emb)

    entity_emb = keras.layers.Dot(axes=[-1,-2])([entity_input,text_vecs]) #(max_entity_length,400)
    entity_vecs = entity_emb

    if entity_type:
        entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs,entity_type_emb])
        entity_vecs = Dense(400)(entity_vecs)
    if entity_cnn:
        entity_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(entity_emb)
    if entity_lstm:
        entity_vecs = Bidirectional(LSTM(200,return_sequences=True))(entity_vecs)
    entity_vecs = keras.layers.Reshape((max_entity_num,400))(entity_vecs)

    entity_vecs1 = keras.layers.TimeDistributed(RepeatVector(max_entity_num))(entity_vecs)
    entity_vecs2 = Reshape((max_entity_num*400,))(entity_vecs)
    entity_vecs2 = RepeatVector(max_entity_num)(entity_vecs2)
    entity_vecs2 = Reshape((max_entity_num,max_entity_num,400,))(entity_vecs2)

    entity_vecs0 = keras.layers.Concatenate(axis=-1)([entity_vecs1,entity_vecs2])

    knowledge_embedding_layer = Embedding(10,dim_knowledge,trainable=True)
    knowledge_vecs = knowledge_embedding_layer(knowledge_input)
    knowledge_vecs = Dropout(0.3)(knowledge_vecs)
    dim = 800
    if use_knowledge:
        entity_vecs0 = keras.layers.Concatenate(axis=-1)([entity_vecs0,knowledge_vecs])
        dim += dim_knowledge
   # entity_vecs0 = knowledge_vecs
    #entity_vecs0 = knowledge_vecs
    classifier = get_classifier(dim)
    pred = keras.layers.TimeDistributed(keras.layers.TimeDistributed(classifier))(entity_vecs0)
    pred = Reshape((max_entity_num*max_entity_num,1))(pred)


    return keras.Model([texts_input,entity_input,entity_type_input,bert_text_input,bert_seg_input,knowledge_input],pred)

def get_modelv2(emb_matrix,bert,tp_dict,use_bert,char_cnn,char_lstm,entity_type,entity_cnn,entity_lstm):
    embedding_matrix = emb_matrix
    texts_input = Input(shape=(max_text_length,),dtype='int32')
    entity_input = Input(shape=(max_entity_num,max_text_length),dtype='float32')
    entity_type_input = Input(shape=(max_entity_num,),dtype='int32')
    bert_text_input = Input(shape=(max_text_length,),dtype='int32')
    bert_seg_input = Input(shape=(max_text_length,),dtype='int32')

    embedding_layer = Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=True)
    text_embedding = embedding_layer(texts_input)
    bert_embedding = bert([bert_text_input,bert_seg_input])
    bert_embedding = RemoveMask()(bert_embedding)
    
    if use_bert:
        text_embedding = keras.layers.Concatenate(axis=-1)([text_embedding,bert_embedding])
        text_embedding = Dropout(0.2)(text_embedding)
        text_embedding = Dense(400)(text_embedding)
        
    
    text_vecs = text_embedding
    if char_cnn:
        text_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(text_vecs)
    if char_lstm:
        text_vecs = Bidirectional(LSTM(200,return_sequences=True))(text_vecs) #(max_text_length,400)
    
    entity_type_embedding_layer = Embedding(len(tp_dict),200,trainable=True)
    entity_type_emb = entity_type_embedding_layer(entity_type_input)
    entity_type_emb = Dropout(0.2)(entity_type_emb)

    entity_emb = keras.layers.Dot(axes=[-1,-2])([entity_input,text_vecs]) #(max_entity_length,400)
    entity_vecs = entity_emb

    if entity_type:
        entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs,entity_type_emb])
        entity_vecs = Dense(400)(entity_vecs)
    if entity_cnn:
        entity_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(entity_emb)
    if entity_lstm:
        entity_vecs = Bidirectional(LSTM(200,return_sequences=True))(entity_vecs)
    entity_vecs = keras.layers.Reshape((max_entity_num,400))(entity_vecs)

    entity_vecs1 = keras.layers.TimeDistributed(RepeatVector(max_entity_num))(entity_vecs)
    entity_vecs2 = Reshape((max_entity_num*400,))(entity_vecs)
    entity_vecs2 = RepeatVector(max_entity_num)(entity_vecs2)
    entity_vecs2 = Reshape((max_entity_num,max_entity_num,400,))(entity_vecs2)

    entity_vecs0 = keras.layers.Concatenate(axis=-1)([entity_vecs1,entity_vecs2])

    dim = 800

    classifier = get_classifier(dim)
    pred = keras.layers.TimeDistributed(keras.layers.TimeDistributed(classifier))(entity_vecs0)
    pred = Reshape((max_entity_num*max_entity_num,1))(pred)


    return keras.Model([texts_input,entity_input,entity_type_input,bert_text_input,bert_seg_input],pred)