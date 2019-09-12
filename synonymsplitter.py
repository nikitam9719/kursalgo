#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import coocur
import sys


#f1=codecs.open('synmaster.txt', 'r',encoding='mbcs')
#f1=open('synmaster.txt', 'r')
#f2=open('synmasternew.txt', 'w')
#f2=codecs.open('synmasternew.txt', 'w',encoding='utf-8')
#f4=open('russian_newsvocab.txt','w')



def main():
    f1=open(sys.argv[1], 'r')
    f2=open(sys.argv[2], 'w')
    for l in f1:
        a=l.split('\n')[0].split('\r')[0].split('|')
        if(' ' in a[0]):
            continue
        else:
            i=0
            k=a[0]
            while i<len(a)-1:
                i=i+1
                if((' ' in a[i])or (len(a[i])==0)):
                    continue
                else:
                    f2.write(k+'\t'+a[i]+'\n')


if __name__ == '__main__':
    #preprocess_text(f3)
    main()
    f3 = open(sys.argv[3], 'r')
    a=coocur.build_vocab(f3)
    f4=open(sys.argv[4], 'w')
    for i in a:
        f4.write( i+' '+str(a[i][1])+'\n')
    f6=open(sys.argv[3],'r')
    temp=coocur.build_cooccur(a, f6, 5, min_count=20)
    #print temp
    #print a

