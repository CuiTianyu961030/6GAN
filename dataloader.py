import numpy as np
import ipaddress


class DataProcessing(object):
    def __init__(self, data_path, vocab_path, num_data):
        self.data_file = data_path
        self.vocab_file = vocab_path
        self.ip_split_list = []
        self.vocab_list = []
        self.conjunction = ""
        self.num_data = num_data

    def create_work_data(self, source_file, work_file):
        f = open(source_file, 'r')
        data = f.readlines()
        f.close()
        f = open(work_file, 'w')
        f.writelines(data[:self.num_data])
        f.close()

    def ip_split(self, source_file):
        self.ip_split_list = []
        f = open(source_file, 'r')
        for address in f:
            address = ipaddress.ip_address(address[:-1]).exploded
            nybbles = address.split(":")
            self.ip_split_list.append(self.conjunction.join(nybbles))
        f.close()
        f = open(self.data_file, 'w')
        self.ip_split_list = [i + '\n' for i in self.ip_split_list][:self.num_data]
        f.writelines(self.ip_split_list)
        f.close()

    def gen_vocab(self):
        self.vocab_list = ['<PAD>', '<EOS>', '<GO>']
        self.vocab_list.extend([str(hex(i))[-1] for i in range(16)])
        f = open(self.vocab_file, 'w')
        self.vocab_list = [i + '\n' for i in self.vocab_list]
        f.writelines(self.vocab_list)
        f.close()


class GenDataLoader(object):
    def __init__(self, batch_size, vocab_dict):
        self.batch_size = batch_size
        self.token_stream = []
        self.vocab_size = 0
        self.vocab_dict = vocab_dict

    def create_batches(self, data_file_list):
        """make self.token_stream into a integer stream."""
        self.token_stream = []
        print("load %s file data.." % ' '.join(data_file_list))
        for data_file in data_file_list:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # cut the taken_stream's length exactly equal to num_batch * batch_size
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        print("      Load %d * %d batches" % (self.num_batch, self.batch_size))

    def next_batch(self):
        """take next batch by self.pointer"""
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataLoader(object):
    def __init__(self, batch_size, vocab_dict, max_sequence_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.vocab_dict = vocab_dict
        self.max_sequence_length = max_sequence_length

    def load_train_data(self, positive_file_list, negative_file_list):
        # Load data
        positive_examples = []
        negative_examples = []
        for positive_file in positive_file_list:
            class_examples = []
            with open(positive_file)as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    class_examples .append(parse_line)
            positive_examples.append(class_examples)
        for negative_file in negative_file_list:
            with open(negative_file)as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    negative_examples.append(parse_line)

        examples = []
        for class_examples in positive_examples:
            examples = examples + class_examples
        examples = examples + negative_examples
        self.sentences = np.array(examples)
        self.sentences = self.padding(self.sentences, self.max_sequence_length)

        # Generate labels
        positive_class_num = len(positive_file_list) + 1
        onehot_labels = np.zeros((positive_class_num, positive_class_num))
        for i in range(positive_class_num):
            onehot_labels[i][i] = 1
        count = 0
        for label, class_examples in zip(onehot_labels[:-1], positive_examples):
            if count == 0:
                self.labels = [label for _ in class_examples]
            else:
                self.labels = np.concatenate([self.labels, [label for _ in class_examples]], 0)
            count += 1

        negative_labels = [onehot_labels[-1] for _ in negative_examples]
        self.labels = np.concatenate([self.labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        """take next batch (sentence, label) by self.pointer"""
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def padding(self, inputs, max_sequence_length):
        batch_size = len(inputs)
        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        return inputs_batch_major


class PrefixLoader(object):
    def __init__(self, aliased_prefix_file):
        self.aliased_prefix_file = aliased_prefix_file
        self.prefix_data = []

    def load_prefixes(self):
        f = open(self.aliased_prefix_file, 'r')
        aliased_prefixes = f.readlines()
        f.close()

        for aliased_prefix in aliased_prefixes:
            prefix, prefix_len = aliased_prefix.split('/')
            prefix = ipaddress.ip_address(prefix).exploded
            prefix = ''.join(prefix.split(':'))[:int(int(prefix_len[:-1]) / 4)]
            self.prefix_data.append(prefix)
        return self.prefix_data
