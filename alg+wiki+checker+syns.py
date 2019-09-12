import learn as learn
import opencorpora
from ast import literal_eval

import numpy
import pycrfsuite
import scipy
import sklearn
import sklearn_crfsuite
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import sklearn.metrics

syndict1 = dict()
syndict2 = dict()

def make_dict(a):
    d=dict()
    with open(a,"r",encoding="utf-16") as f:
        for i in f.readlines():
            if(len(i)<3):
                continue
            k=i.split('/')
            if('Person' in k[1]):
                l=k[0].split(',')
                if(len(l)<2):
                    l=l[0].split('_',1)
                    if(len(l)<2):
                        d.update(((l[0], ''),))
                        continue
                d.update(((l[0], l[1]),))
            else:
                d.update(((k[0], k[1]),))
    return d

def in_dict(word):
    a=dictianory.get(word)
    if(a!=None):
        if("Place" in a):
            return "Place"
        elif("Organisation" in a):
            return "Organisation"
        # elif("Misc" in a):
        #     return "Misc"
        else:
            return "Person"
    else:
        return "S"

def correction_sent(sent):
    counter=-1
    for word in sent:
        counter=counter+1
        mark=0
        wordl=word[0]
        a = dictianory.get(wordl)
        if(a!=None):
            continue
        else:
            for i in range(len(wordl)):
                for letter in list(map(chr, range(97, 123))):
                    test=list(wordl)
                    test.insert(i,letter)
                    if(dictianory.get(''.join(test))!=None):
                        correctedword=test
                        mark=mark+1
                        if(mark>1):
                            break
            if(mark>1):
                continue
            for i in range(len(wordl)):
                test=list(wordl)
                test.pop(i)
                if(dictianory.get(''.join(test))!=None):
                    correctedword=test
                    mark=mark+1
                    if (mark > 1):
                        break
            if(mark>1):
                continue
            for i in range(len(wordl)):
                for letter in list(map(chr, range(97, 123))):
                    test=list(wordl)
                    test.pop(i)
                    test.insert(i,letter)
                    if(dictianory.get(''.join(test))!=None):
                        correctedword=test
                        mark=mark+1
                        if(mark>1):
                            break
            if(mark>1):
                continue
            elif mark==1:
                sent[counter][0]=''.join(correctedword)
    return sent
def full_name_builder(l,sent,index):
    counter=index-1
    curword=sent[index][0]
    full_name=curword
    dictname=dictianory.get(curword)
    dictname=dictname.split('_')
    while counter>=0:
        anotherword=sent[counter][0]
        if('.'in anotherword):
            if(len(anotherword)<2):
                break
            anotherword=anotherword.split('.')[0]
        if(anotherword in dictname):
            full_name=full_name+anotherword
            counter=counter-1
        else:
            break
    leftborder = counter
    counter=index+1
    while counter < len(sent):
        anotherword = sent[counter][0]
        if ('.' in anotherword):
            if(len(anotherword)<2):
                break
            anotherword = anotherword.split('.')[0]
        if (anotherword in dictname):
            full_name = full_name + anotherword
            counter = counter + 1
        else:
            break
    rightborder = counter
    i=leftborder+1
    if(rightborder-leftborder==2):
        return l
    while (i>leftborder) and (i<rightborder):
        l[i]=full_name
        i=i+1
    return l


def word2features(sent, i,indictlist,full_name_list,synfeatlist,synnumlist):
    word = sent[i][0]
    postag = sent[i][1]
    indict=indictlist[i]
    synfeat=synfeatlist[i]
    synnum=synnumlist[i]
    features = {
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
    'postag': postag,
    'postag[:2]': postag[:2],
    'in_dict': indict,
    'synfeat': synfeat,
    'synnum': synnum,
    'synfeat': synfeat,
    'synnum': synnum

}
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            'synfeat': synfeatlist[i-1],
            'synnum': synnumlist[i-1]
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            'synfeat': synfeatlist[i + 1],
            'synnum': synnumlist[i + 1]
        })
    else:
        features['EOS'] = True
    if("Person" in indict):
        features.update({
            'in_dict': full_name_list[i]})

    return features
def sent2features(sent):
    indictlist=list()
    full_name_list=list()
    synfeat=list()
    synnum=list()
    sent=correction_sent(sent)
    for i in range (len(sent)):
        full_name_list.append("")

    for i in range(len(sent)):
        type=in_dict(sent[i][0])
        indictlist.append(type)
        if "Person" in type:
            full_name_list=full_name_builder(full_name_list,sent,i)
    for i in range(len(sent)):
        mark=0
        type="S"
        k=syndict1.get(sent[i][0])
        if(k is not None):
            for temp in k:
                type=in_dict(temp[0])
                if type!="S":
                    mark=1
                    synfeat.append(type)
                    synnum.append(temp[1])
                    break

        if mark==0:
            k=syndict2.get(sent[i][0])
            if(k is not None):
                for temp in k:
                    type=in_dict(temp[0])
                    if type!="S":
                        mark=1
                        synfeat.append(type)
                        synnum.append(temp[1])
                        break
        if mark==0:
            synfeat.append(type)
            synnum.append('1.0')
    return [word2features(sent, i,indictlist,full_name_list,synfeat, synnum) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]
if __name__ == '__main__':
    sents=list()
    sents2=list()
    dictianory=make_dict('Output_Packeruni.txt')
    print("ok")

    f1=open('OutputFile','r',encoding="utf-8")
    for l in f1:
        k=l.strip().split('\t')
        if(syndict1.get(k[0]) is not None):
            syndict1.update({k[0]: (syndict1[k[0]]+[[k[1],k[2]],])})
        else:
            syndict1.update({k[0]: [[k[1],k[2]], ]})
        if(syndict2.get(k[1]) is not None):
            syndict2.update({k[1]: (syndict1[k[1]]+[[k[0],k[2]],])})
        else:
            syndict1.update({k[1]: [[k[0],k[2]], ]})
    with open('corpus2',"r",encoding="utf-16") as f:
        for i in f.readlines():
            i.replace('\n','')
            k=eval(i)
            sents.append(k)
    with open('test2.txt',"r",encoding="utf-16") as f:
        for i in f.readlines():
            i.replace('\n','')
            k=eval(i)
            sents2.append(k)
    X_train = [sent2features(s) for s in sents]
    y_train = [sent2labels(s) for s in sents]
    X_test = [sent2features(s) for s in sents2]
    y_test = [sent2labels(s) for s in sents2]
    print(sent2features(sents[0])[0])
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.051508376528013232,
        c2=0.0027557838685648719,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    #labels.remove('O')
    y_pred = crf.predict(X_test)
    #print(cross_val_score(crf,X_train,y_train,cv=5,scoring='f1_micro'))
    print(metrics.flat_f1_score(y_test, y_pred,
        average='weighted'))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
        ))