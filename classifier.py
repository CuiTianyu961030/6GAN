import os
import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

source_file = 'data/source_data/responsive-addresses.work'
source_data = 'data/source_data/responsive-addresses.data'

rfc_profile = 'data/save_data/rfc_profile.txt'
ec_profile = 'data/save_data/ec_profile.txt'
ec_cluster = 'data/save_data/ec_cluster.txt'
ipv62vec_profile = 'data/save_data/ipv62vec_profile.txt'

category_path = 'data/category_data/'
rfc_data_path = category_path + 'rfc/data/'
ec_data_path = category_path + 'ec/data/'
ipv62vec_data_path = category_path + 'ipv62vec/data/'
rfc_id_path = category_path + 'rfc/id/'
ec_id_path = category_path + 'ec/id/'
ipv62vec_id_path = category_path + 'ipv62vec/id/'


class RFCBased(object):
    def __init__(self, batch_size):
        self.classify_dict = {}
        self.cmd = 'cat data/source_data/responsive-addresses.work | ../../Tools/ipv6toolkit/addr6 -i -d > data/save_data/rfc_profile.txt'
        self.rfc_profile = rfc_profile
        self.source_data = source_data
        self.rfc_data_path = rfc_data_path
        self.rfc_id_path = rfc_id_path
        self.batch_size = batch_size
        self.emb_data_num = []

    def create_category(self):
        os.system(self.cmd)
        f = open(self.rfc_profile, 'r')
        g = open(self.source_data, 'r')
        i = 0
        for line, data in zip(f, g):
            label = line.split('=')[3]
            if label not in self.classify_dict.keys():
                self.classify_dict[label] = []
            self.classify_dict[label].append(data)
            i += 1
        g.close()
        f.close()

        for type in self.classify_dict.keys():
            f = open(self.rfc_data_path + type + '.txt', 'w')
            f.writelines(self.classify_dict[type])
            f.close()

    def gen_id_file(self, vocab_dict):
        emb_file_list = []
        for filename in sorted(os.listdir(self.rfc_data_path)):
            emb_id_file = self.rfc_id_path + filename[:-3] + 'id'
            emb_data_file = self.rfc_data_path + filename
            data_num = self.gen_id_data(emb_id_file, emb_data_file, vocab_dict)
            if data_num >= self.batch_size:
                emb_file_list.append(emb_id_file)
            else:
                print('Unload %s, data num %s < batch size' % (emb_id_file, data_num))
        return emb_file_list

    def gen_id_data(self, emb_id_file, emb_data_file, vocab_dict):
        conjunction = ' '
        g = open(emb_id_file, 'w')
        f = open(emb_data_file, 'r')
        data_num = 0
        for data in f:
            id_data = [str(vocab_dict[i]) for i in data[:-1]]
            g.write(conjunction.join(id_data) + '\n')
            data_num += 1
        if data_num >= self.batch_size:
            self.emb_data_num.append(data_num)
        f.close()
        g.close()
        return data_num


class EntropyClustering(object):
    def __init__(self, batch_size, k=4):
        self.classify_dict = {}
        self.classify_prefix_dict = {}
        self.k = k
        self.profile_cmd = 'cat data/source_data/responsive-addresses.data | ' + \
                           '../../Tools/entropy-clustering/profiles > data/save_data/ec_profile.txt'
        self.cluster_cmd = 'cat data/save_data/ec_profile.txt | ' + \
                           '../../Tools/entropy-clustering/clusters -kmeans -k ' \
                           + str(k) + ' > data/save_data/ec_cluster.txt'
        self.ec_profile = ec_profile
        self.ec_cluster = ec_cluster
        self.source_data = source_data
        self.ec_data_path = ec_data_path
        self.ec_id_path = ec_id_path
        self.batch_size = batch_size
        self.emb_data_num = []

    def create_category(self):
        for filename in os.listdir(self.ec_data_path):
            os.remove(self.ec_data_path + filename)
        for filename in os.listdir(self.ec_id_path):
            os.remove(self.ec_id_path + filename)

        # os.system(self.profile_cmd)
        # os.system(self.cluster_cmd)
        for i in range(self.k):
            self.classify_dict[str(i)] = []
            self.classify_prefix_dict[str(i)] = []
        type_pointer = 0
        class_prefix = []
        f = open(self.ec_cluster, 'r')
        for line in f:
            if line[0] == '=':
                class_prefix = []
            elif line[0] == '\n':
                self.classify_prefix_dict[str(type_pointer)].extend(class_prefix)
                type_pointer += 1
            elif line[:7] == 'SUMMARY':
                break
            else:
                class_prefix.append(line[:8])
        f.close()
        print(self.classify_prefix_dict['3'])

        for type in self.classify_prefix_dict.keys():
            for prefix in self.classify_prefix_dict[type]:
                f = open(self.source_data, 'r')
                for line in f:
                    if prefix == line[:8]:
                        self.classify_dict[type].append(line)
                f.close()

        for type in self.classify_dict.keys():
            f = open(self.ec_data_path + 'cluster_' + type + '.txt', 'w')
            f.writelines(self.classify_dict[type])
            f.close()

    def gen_id_file(self, vocab_dict):
        emb_file_list = []
        for filename in sorted(os.listdir(self.ec_data_path)):
            emb_id_file = self.ec_id_path + filename[:-3] + 'id'
            emb_data_file = self.ec_data_path + filename
            data_num = self.gen_id_data(emb_id_file, emb_data_file, vocab_dict)
            if data_num >= self.batch_size:
                emb_file_list.append(emb_id_file)
            else:
                print('Unload %s, data num %s < batch size' % (emb_id_file, data_num))
        return emb_file_list

    def gen_id_data(self, emb_id_file, emb_data_file, vocab_dict):
        conjunction = ' '
        g = open(emb_id_file, 'w')
        f = open(emb_data_file, 'r')
        data_num = 0
        for data in f:
            id_data = [str(vocab_dict[i]) for i in data[:-1]]
            g.write(conjunction.join(id_data) + '\n')
            data_num += 1
        if data_num >= self.batch_size:
            self.emb_data_num.append(data_num)
        f.close()
        g.close()
        return data_num


class IPv62Vec(object):
    def __init__(self, batch_size):
        self.source_data = source_data
        self.ipv62vec_profile = ipv62vec_profile
        self.ipv62vec_data_path = ipv62vec_data_path
        self.ipv62vec_id_path = ipv62vec_id_path
        self.batch_size = batch_size
        self.emb_data_num = []

    def create_category(self):
        for filename in os.listdir(self.ipv62vec_data_path):
            os.remove(self.ipv62vec_data_path + filename)
        for filename in os.listdir(self.ipv62vec_id_path):
            os.remove(self.ipv62vec_id_path + filename)

        f = open(self.source_data, 'r')
        raw_data = f.readlines()
        f.close()

        index_alpha = '0123456789abcdefghijklmnopqrstuv'
        address_sentences = []
        for address in raw_data:
            address_words = []
            for nybble, index in zip(address[:-1], index_alpha):
                address_words.append(nybble + index)
            address_sentences.append(" ".join(address_words) + '\n')

        f = open(self.ipv62vec_profile, 'w')
        f.writelines(address_sentences)
        f.close()

        sentences = word2vec.LineSentence(self.ipv62vec_profile)
        model = word2vec.Word2Vec(sentences, alpha=0.025, min_count=0, size=100, window=5,
                                  sg=0, hs=0, negative=5, ns_exponent=0.75, iter=5)

        vocab = list(model.wv.vocab.keys())
        X_tsne = TSNE(n_components=2, learning_rate=200, perplexity=30).fit_transform(model.wv[vocab])

        address_split_sentences = [sentence[:-1].split(' ') for sentence in address_sentences]
        x = []
        y = []
        for address_split_sentence in address_split_sentences:
            x_one_sample = []
            y_one_sample = []
            for word in address_split_sentence:
                x_one_sample.append(X_tsne[vocab.index(word), 0])
                y_one_sample.append(X_tsne[vocab.index(word), 1])
            x.append(np.mean(x_one_sample))
            y.append(np.mean(y_one_sample))

        dim_reduced_data = []
        for i, j in zip(x, y):
            dim_reduced_data.append([i, j])
        dim_reduced_data = pd.DataFrame(dim_reduced_data)
        dim_reduced_data.columns = ['x', 'y']

        data = self.cluster(dim_reduced_data)
        self.search_cluster(data, raw_data)

    def cluster(self, data):
        db = DBSCAN(eps=0.0085, min_samples=self.batch_size).fit(data)
        data['labels'] = db.labels_
        n_clusters_ = 0
        try:
            n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            if n_clusters_ > 10:
                raise ClassNumError(n_clusters_)
        except ClassNumError as e:
            print('ClassNumError: generated class num %s > 10, please reset IPv62Vec parameters.' % e.n_clusters)
            os._exit(0)
        print('cluster num', n_clusters_)
        return data

    def search_cluster(self, data, raw_data):
        index_list = []
        labels = set(data['labels'])
        for label in labels:
            index_list.append(data[data['labels'] == label].index)

        for index, label in zip(index_list, labels):
            f = open(self.ipv62vec_data_path + 'cluster_' + str(label) + '.txt', 'w')
            for i in index:
                f.write(raw_data[i])
            f.close()

    def gen_id_file(self, vocab_dict):
        emb_file_list = []
        for filename in sorted(os.listdir(self.ipv62vec_data_path)):
            emb_id_file = self.ipv62vec_id_path + filename[:-3] + 'id'
            emb_data_file = self.ipv62vec_data_path + filename
            data_num = self.gen_id_data(emb_id_file, emb_data_file, vocab_dict)
            if data_num >= self.batch_size:
                emb_file_list.append(emb_id_file)
            else:
                print('Unload %s, data num %s < batch size' % (emb_id_file, data_num))
        return emb_file_list

    def gen_id_data(self, emb_id_file, emb_data_file, vocab_dict):
        conjunction = ' '
        g = open(emb_id_file, 'w')
        f = open(emb_data_file, 'r')
        data_num = 0
        for data in f:
            id_data = [str(vocab_dict[i]) for i in data[:-1]]
            g.write(conjunction.join(id_data) + '\n')
            data_num += 1
        if data_num >= self.batch_size:
            self.emb_data_num.append(data_num)
        f.close()
        g.close()
        return data_num


class ClassNumError(Exception):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
