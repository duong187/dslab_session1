from nltk.stem.porter import PorterStemmer
import os,sys
import re
import collections
import numpy as np
def gather_20newsgroup_data():
    path= 'D:/Python/machine_learning/tf_idf/20news-bydate/'
    dirs=[path+dir_name+'/' for dir_name in os.listdir(path) if not os.path.isfile(path+dir_name)]
    train_dir,test_dir=(dirs[0],dirs[1]) if 'train' in dirs[0] else (dirs[1],dirs[0])
    newsgroup_list=[newsgroup for newsgroup in os.listdir(train_dir)]
    newsgroup_list.sort()
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/stop_words.txt') as f:
        stop_words=[f.read().splitlines()]
    stemmer=PorterStemmer()
    def collect_data_from(parents_dirs,newsgroup_list):
        data=[]
        for group_id,group_name in enumerate(newsgroup_list):
            label=group_id
            group_path=parents_dirs+group_name+'/'
            file_list=[(file_name,group_path+file_name) for file_name in os.listdir(group_path) if os.path.isfile(group_path+file_name)]
            file_list.sort()
            for file_name,file_path in file_list:
                with open(file_path) as f:
                    text=f.read().lower()
                    words=[stemmer.stem(word) for word in re.split('\W+',text) if word not in stop_words]
                    content=' '.join(words)
                    assert len(content.splitlines())==1
                    data.append(str(label)+'<fff>'+file_name+'<fff>'+content)
        return data
    train_data=collect_data_from(parents_dirs=train_dir,newsgroup_list=newsgroup_list)
    test_data=collect_data_from(parents_dirs=test_dir,newsgroup_list=newsgroup_list)
    full_data=train_data+test_data
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/20news-bydate-train-procceed.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/20news-bydate-test-procceed.txt','w') as f:
        f.write('\n'.join(test_data))
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/20news-bydate-full-procceed.txt','w') as f:
        f.write('\n'.join(full_data))



def generate_vocabulary(data_path):
    def compute_idf(df,corpus_size):
        assert df>0
        return np.log10(corpus_size*1./df)
    with open(data_path)as f:
        lines=f.read().splitlines()
        corpus_size=len(lines)
        doc_count=collections.defaultdict(int)
        for line in lines:
            feature=line.split('<fff>')
            text=feature[-1]
            words=list(set(text.split()))
            for word in words:
                doc_count[word]+=1
    word_idf=[(word,compute_idf(df,corpus_size)) for word,df in zip(doc_count.keys(),doc_count.values()) if df>10 and not word.isdigit()]
    word_idf.sort(key=lambda word_idf:-word_idf[1])
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/word_idf.txt','w') as f:
        f.write('\n'.join([word+'<fff>'+str(idf) for word,idf in word_idf]))

def get_tf_idf(data_path):
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/word_idf.txt') as f:
        words_idfs=[(line.split('<fff>')[0],float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        words_IDs=dict([word,index] for index,(word,idf) in enumerate(words_idfs))
        idfs=dict(words_idfs)
        
    with open (data_path) as f:
        documents=[(int(line.split('<fff>')[0]),int(line.split('<fff>')[1]),line.split('<fff>')[2])
        for line in f.read().splitlines()]
        data_tf_idf=[]
    for document in documents:
        label,word_id,text=document
        words=[word for word in text.split()if word in idfs]
        words_set=list(set(words))
        max_term_freq=max(words.count(word)for word in words_set)
        sum_square=0.0
        words_tf_idf=[]
        for word in words_set:
            freq=words.count(word)
            tf_idf_value=freq*1./max_term_freq*idfs[word]
            sum_square+=tf_idf_value**2
            words_tf_idf.append((words_IDs[word],tf_idf_value))
        tf_idf_normalized=[(str(index)+':'+str(value*1./sum_square))for index,value in words_tf_idf]
        sparse_rep=' '.join(tf_idf_normalized)
        data_tf_idf.append((label,word_id,sparse_rep))
    with open('D:/Python/machine_learning/tf_idf/20news-bydate/data_tf_idf.txt','w') as f:
        f.write('\n'.join([str(label)+'<fff>'+str(word_id)+'<fff>'+sparse_rep for label,word_id,sparse_rep in data_tf_idf]))
get_tf_idf('D:/Python/machine_learning/tf_idf/20news-bydate/20news-bydate-train-procceed.txt')








