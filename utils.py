# -*- coding: utf-8 -*-
import numpy as np
import operator
import pandas as pd
import pymorphy2
import re
import json
import os
from random import shuffle

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk import pos_tag

stop_words = stopwords.words('english')
stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer

from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as features

nlu = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='af56f953-5b88-4254-8ed1-e0f408ae8093',
    password='5TWZgCWZFgaM')

DIR = 'data/'

tf_idf_basis_dict = {}
tf_idf_themes_dict = {'n': [], # NOUN
                      'v': [], # VERB
                      'a': [], # ADJ
                      'r': [] } # ADV

def fill_dict(tf_idf_basis, tf_idf_themes, basis_soa, themes_soa):
    for word in tf_idf_basis:
        for soa in basis_soa:
            sntc = soa['sentence']
            if sntc.find(word) != -1:
                s_pos = get_wn_tagged_sentence(sntc)
                w_pos = s_pos[[i for i, w in enumerate(s_pos) if word in w[0]][0]][1]
                tf_idf_basis_dict[word] = w_pos
    for word in tf_idf_themes:
        for soa in themes_soa:
            sntc = soa['sentence']
            if sntc.find(word) != -1:
                s_pos = get_wn_tagged_sentence(sntc)
                w_pos = s_pos[[i for i, w in enumerate(s_pos) if word in w[0]][0]][1]
                if w_pos in tf_idf_themes_dict:
                    tf_idf_themes_dict[w_pos].append(word)

def replace_tf_idf_top(tf_idf_basis, tf_idf_themes, basis_soa, themes_soa):
    word_new_word_dict = {}
    for soa in basis_soa:
        for word in tf_idf_basis:
            sntc = soa['sentence']
            if sntc.find(word) != -1:
                w_pos = tf_idf_basis_dict[word]
                pos_arr = tf_idf_themes_dict[w_pos][:]
                shuffle(pos_arr)
                try:
                    new_word = pos_arr[0]
                    word_new_word_dict[word] = new_word
                except:
                    pass
    return word_new_word_dict
                

def get_synonyms(word):
    syn_arr = [st.lemma_names()[0] for st in wn.synsets(word)]
    syn_arr = filter(lambda x: x != word, syn_arr)
    return list(set(syn_arr))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return ''#wn.NOUN#''
    
def get_wn_tagged_sentence(sentence):
    wt = word_tokenize(sentence)
    tagged_sentence = pos_tag(wt)
    return [(word, get_wordnet_pos(tag)) for word, tag in tagged_sentence]

def get_semantic_roles(text, limit=500, entities=True, keywords=None):
    r = nlu.analyze(text=text,
                    features=[features.SemanticRoles(limit=limit, 
                                                     entities=entities, 
                                                     keywords=keywords)])
    return r['semantic_roles']

def prepare_txt(texts):
    for text in texts:
        with open(text, encoding="utf8") as f:
            f_new = re.sub(r'\u2022.*?\\n', '', f.read()).replace('\\n', '\n')
        with open(text, 'w', encoding="utf8") as f:
            f.write(f_new)
            
# sort - less lenght on start
def get_sentences(texts, min_len=5, sort=False):
    rows = []
    for text in texts:
        with open(text, encoding="utf8") as f:
            rows += sent_tokenize(f.read())
    return [s for s in (sorted(rows, key=len) if sort else rows) if len(s) > min_len]

# should recode this
def get_sentences_for_watson(sentences):
    parts = []
    part = sentences[0]
    magic_number = 9000
    for i in range(1, len(sentences)):
        part = ' '.join([part, sentences[i]])
        if len(part) > magic_number:
            parts.append(part)
            part = ''
    if len(part) > 10:
        parts.append(part)
    return parts

def write_soa_json(snts, fname):
    soa = []
    for part in snts:
        soa += get_semantic_roles(part)
    with open(DIR+'soa_json/'+fname, 'w') as f:
        json.dump(soa, f)
    return soa

def load_soa(snts, fname, rewrite=False):
    soa = []
    if os.path.exists(DIR+'soa_json/'+fname):
        if rewrite:
            soa = write_soa_json(snts, fname)
        else:
            with open(DIR+'soa_json/'+fname) as f:
                soa = json.load(f)
    else:
        soa = write_soa_json(snts, fname)
    return soa

def filt_soa_json(soas, max_words_len=7, min_words_len=3):
    new_soas = []
    for soa in soas:
        s, o, a = '', '', ''
        if 'subject' in soa:
            s = soa['subject']['text']
        if 'object' in soa:
            o = soa['object']['text']
        if 'action' in soa:
            a = soa['action']['text']
        s_words_count = len(word_tokenize(s))
        o_words_count = len(word_tokenize(o))
        a_words_count = len(word_tokenize(a))
        sent_words_count = len(word_tokenize(soa['sentence']))
        left_words_count = sent_words_count - (s_words_count + o_words_count + a_words_count)
        if max(s_words_count, o_words_count, a_words_count) < max_words_len + 1:
            if left_words_count < max_words_len + 1 and left_words_count > min_words_len - 1:
                new_soas.append(soa)
    return new_soas

def get_sbj_obj_action(soa):
    sbj = {'text': '', 'entity': ''}
    obj = {'text': '', 'entity': ''}
    action = {'text': '', 'tense': ''}
    if 'subject' in soa:
        s = soa['subject']
        if 'entities' in s:
            if len(s['entities']) != 0:
                sbj['entity'] = s['entities'][0]['type']
        sbj['text'] = s['text']
    if 'object' in soa:
        o = soa['object']
        if 'entities' in o:
            if len(o['entities']) != 0:
                obj['entity'] = o['entities'][0]['type']
        obj['text'] = o['text']
    if 'action' in soa:
        a = soa['action']
        action['tense'] = a['verb']['tense']
        action['text'] = a['text']
    return sbj, obj, action

def get_by_sbj_obj_tense(soas, s_entity, o_entity, tense):
    for soa in soas:
        if 'action' in soa and 'subject' in soa and 'object' in soa:
            if 'entities' in soa['subject'] and 'entities' in soa['object'] \
            and len(soa['subject']['entities']) != 0 and len(soa['object']['entities']) != 0:
                new_s_entity = soa['subject']['entities'][0]['type']
                new_o_entity = soa['object']['entities'][0]['type']
                new_tense = soa['action']['verb']['tense']
                if s_entity == new_s_entity and o_entity == new_o_entity and tense == new_tense:
                    print('got by sbj, obj, tense')
                    return soa['subject']['text'], soa['object']['text'], soa['action']['text']
    return '', '', ''

def get_by_sbj_tense(soas, s_entity, tense):
    for soa in soas:
        if 'action' in soa and 'subject' in soa:
            if 'entities' in soa['subject'] and len(soa['subject']['entities']) != 0:
                new_s_entity = soa['subject']['entities'][0]['type']
                new_tense = soa['action']['verb']['tense']
                if s_entity == new_s_entity and tense == new_tense:
                    print('got by sbj, tense')
                    obj = ''
                    if 'object' in soa: obj = soa['object']['text']   
                    return soa['subject']['text'], obj, soa['action']['text'] 
    return '', '', ''

def get_by_obj_tense(soas, o_entity, tense):
    for soa in soas:
        if 'action' in soa and 'object' in soa:
            if 'entities' in soa['object'] and len(soa['object']['entities']) != 0:
                new_o_entity = soa['object']['entities'][0]['type']
                new_tense = soa['action']['verb']['tense']
                if o_entity == new_o_entity and tense == new_tense:
                    print('got by obj, tense')
                    sbj = ''
                    if 'subject' in soa: sbj = soa['subject']['text']   
                    return sbj, soa['object']['text'], soa['action']['text'] 
    return '', '', ''

def get_by_tense(soas, tense):
    for soa in soas:
        if 'action' in soa:
            new_tense = soa['action']['verb']['tense']
            if tense == new_tense:
                print('got by tense')
                sbj = ''
                obj = ''
                if 'subject' in soa: 
                    sbj = soa['subject']['text']  
                if 'object' in soa:
                    obj = soa['object']['text'] 
                return sbj, obj, soa['action']['text']
    return '', '', ''

def get_by_random(soas):
    shuffle(soas)
    sbj, obj, action = get_sbj_obj_action(soas[0])
    return sbj['text'], obj['text'], action['text']

# Change below so that it would look for the best subject and object. Now it looks for same entity either in subject or in object. If none is found, it gets everything randomly.
#refactor
def get_themed_soa(soas, s_entity, o_entity, tense):
    sbj = ''
    obj = ''
    action = ''
    if s_entity != '' and o_entity != '' and tense != '':
        sbj, obj, action = get_by_sbj_obj_tense(soas, s_entity, o_entity, tense)
        if sbj != '': return sbj, obj, action
    if s_entity != '' and tense != '':
        sbj, obj, action = get_by_sbj_tense(soas, s_entity, tense)
        if sbj != '': return sbj, obj, action
    if o_entity != '' and tense != '':
        sbj, obj, action = get_by_obj_tense(soas, o_entity, tense)
        if obj != '': return sbj, obj, action
    if tense != '':
        sbj, obj, action = get_by_tense(soas, tense)
        if tense != '': return sbj, obj, action
    return get_by_random(soas)

def get_sentences_to_show(basis_soa=None, themes_soa=None, n=10, shuffle_themes=True):
    if basis_soa == None or themes_soa == None:
        print('give basis_soa and themes_soa, please')
        return None
    themes_soa = themes_soa[:]
    if shuffle_themes:
        shuffle(themes_soa)
    sentences_to_show = []
    for i in range(n):
        soa = basis_soa[i]
        org_sbj, org_obj, org_action = get_sbj_obj_action(soa)

        new_sbj, new_obj, new_action = get_themed_soa(themes_soa, org_sbj['entity'], org_obj['entity'], org_action['tense'])
        sentence = soa['sentence']

        if org_sbj['text'] != '' and new_sbj != '':
            sentence = sentence.replace(org_sbj['text'], new_sbj)
        if org_obj['text'] != '' and new_obj != '':
            sentence = sentence.replace(org_obj['text'], new_obj)
        if org_action['text'] != '' and new_action != '':
            sentence = sentence.replace(org_action['text'], new_action)

        sentences_to_show.append(sentence)
    return sentences_to_show

def removePunctuation(text):
    return re.sub(r'[^a-zA-Zа-яА-Я0-9 ]', ' ', text.lower()).replace('  ',' ').strip()

def preproc(sentence,filter_words=False):
    if filter_words:
        return [x for x in removePunctuation(
            sentence
        ).split(' ') \
                if not (x in stop_words)
                ]
    else:
        return [x for x in removePunctuation(
            sentence
        ).split(' ') ]

def stem(sent):
    return [stemmer.stem(word) for word in sent]

def get_TfIdf_scores(txt_list, ngram_range=(1, 1), return_idf=False, stemm=True, n_threads=3, filter_words=False):
    """
    txt_list - list of texts (words separated by ' ')
    return_idf=True => returs a tuple of (tfidf scores for doc, idf scores global)
    """

    assert ((type(txt_list[0]) == str ) and (len(txt_list) > 1)), "invalid input"
    
    txt = [x for x in map(lambda x: preproc(x,filter_words), txt_list)]
    
    """if filter_words:
        txt = [list(filter(lambda z: morph.parse(z)[0].tag.POS in ['NOUN', 'VERB', 'ADJF', 'ADJS'],
                      x)) for x in txt]  # filtering useless words"""
    if filter_words:
        txt = [list(filter(lambda z: nltk.pos_tag([z])[0][1][0] in ['N', 'V', 'A'],
                      x)) for x in txt]  # filtering useless words"""

    if stemm:
        #new_data = [x for x in Pool(n_threads).imap_unordered(stem, txt)]
        new_data = [stem(x) for x in txt]
    else:
        new_data = txt

    all_words = [item for sublist in new_data for item in sublist]

    unique_words = list(set(all_words))
    un_word_size = len(unique_words)

    def word_freq(w):
        return all_words.count(w) / un_word_size

    frequency_l = map(word_freq, unique_words)
    #   frequency_l = Pool(n_threads).imap_unordered(word_freq,unique_words)
    frequency = dict(zip(unique_words, frequency_l))

    # print(frequency)

    sorted_f = sorted(frequency.items(),
                      key=operator.itemgetter(1),
                      reverse=True)
    str_freq = [
        (w, f) for w, f in sorted_f if not (w in stopwords.words('english'))
        ]

    ready2 = [' '.join(q) for q in new_data if len(q) > 0]

    vectorizer = TfidfVectorizer(
                                 max_features=len(unique_words),
                                 ngram_range=ngram_range,
                                 use_idf=True,encoding=u'utf-8',decode_error='ignore')

    X = vectorizer.fit_transform(ready2)

    # print("n_samples: %d, n_features: %d" % X.shape)

    feat_names = vectorizer.get_feature_names()

    # print(
    # feat_index = X[needed_text, :].nonzero()[1]

    lst = list(zip(feat_names,np.squeeze(np.asarray(np.sum(X,axis=0)/(X != 0).sum(0)))))

    lst.sort(key=lambda x: x[1], reverse=True)

    res1 = pd.DataFrame(dict(lst), index=['TF/IDF']).T.reset_index()

    res1.columns = ['keyword', 'TF/IDF']

    res2 = pd.DataFrame(np.array([vectorizer.get_feature_names(), vectorizer.idf_])).T
    res2.columns = ['keyword', 'IDF']

    res2 = res2[res2.keyword.isin(stop_words) == False]

    def find_substrings(series,kw):
        return ((series.str.split(' ').apply(lambda x: set(x).issubset(kw.split(' '))))&(series != kw))

    def filter_keywords(df):
        df_l = df.keyword.str.split(' ').apply(len)
        to_drop = df_l.isnull()
        
        for ln in range(df_l.max(),0,-1):
            for i in df.keyword[df_l > ln]:
                to_drop = (to_drop | find_substrings(df.keyword,i))
        return to_drop

    if (not return_idf):
        res = (pd.merge(res2, res1).sort_values(by=['IDF', 'TF/IDF'], ascending=False))
        return res.groupby(['IDF',
                            'TF/IDF']).apply(\
                            lambda x: x.keyword[filter_keywords(x)==False]
                            ).reset_index().drop('level_2',axis=1)
    else:
        return (res1.sort_values(by='TF/IDF'), res2.sort_values(by='IDF'))