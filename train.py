import numpy as np
import tensorflow as tf
import os
import random
from dataloader import GenDataLoader, DisDataLoader, DataProcessing, PrefixLoader
from classifier import RFCBased, EntropyClustering, IPv62Vec
import pickle
from generator import Generator
from discriminator import Discriminator
# from rollout import ROLLOUT

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 200  # embedding dimension 200
HIDDEN_DIM = 200  # hidden state dimension of lstm cell 200
MAX_SEQ_LENGTH = 33  # max sequence length
BATCH_SIZE = 64


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 800
TOTAL_GENERATION = 51200
CLASSIFICATION_METHOD = 0  # 0-rfc, 1-ec, 2-ipv62vec, -1-none
ALIAS_DETECTION = 0

dataset_path = "data/source_data/"
source_file = dataset_path + "responsive-addresses.txt"
work_file = dataset_path + "responsive-addresses.work"
emb_data_file = dataset_path + "responsive-addresses.data"
emb_dict_file = dataset_path + "responsive-addresses.vocab"
emb_id_file = dataset_path + "responsive-addresses.id"
aliased_prefix_file = dataset_path + "aliased-prefixes.txt"

save_path = "data/save_data/"
log_file = save_path + "train.log"
eval_file = save_path + "eval_file.txt"
eval_text_file = save_path + "eval_text_file.txt"

candidate_path = "data/candidate_set/"
model_path = "models/"


def generate_samples(sess, trainable_model, generated_num, output_file, vocab_list, if_log=False, epoch=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num)):
        generated_samples.extend(trainable_model.generate(sess))

    if if_log:
        mode = 'a'
        if epoch == 0:
            mode = 'w'
        with open(eval_text_file, mode) as fout:
            # id_str = 'epoch:%d ' % epoch
            for poem in generated_samples:
                poem = list(poem)
                if 1 in poem:
                    poem = poem[:poem.index(1)]
                buffer = ' '.join([vocab_list[x] for x in poem]) + '\n'
                fout.write(buffer)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            poem = list(poem)
            if 1 in poem:
                poem = poem[:poem.index(1)]
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_infer(sess, trainable_model, epoch, vocab_list, generator_id):
    generated_samples = []
    for _ in range(int(TOTAL_GENERATION / BATCH_SIZE)):
        generated_samples.extend(trainable_model.generate(sess))
    file = candidate_path + 'candidate_generator_' + str(generator_id) + '_epoch_' + str(epoch) + '.txt'
    target_generation = []
    for address in generated_samples:
        address = list(address)
        if 1 in address:
            address = address[:address.index(1)]
        count = 0
        predict_address_str = ""
        for i in address[:-1]:
            predict_address_str += vocab_list[i]
            count += 1
            if count % 4 == 0 and count != 32:
                predict_address_str += ":"
        target_generation.append(predict_address_str + '\n')
    fout = open(file, 'w')
    fout.writelines(list(set(target_generation)))
    fout.close()
    print("%s saves" % file)
    return


def produce_samples(generated_samples):
    produces_sample = []
    for poem in generated_samples:
        poem_list = []
        for ii in poem:
            if ii == 0:  # _PAD
                continue
            if ii == 1:  # _EOS
                break
            poem_list.append(ii)
        produces_sample.append(poem_list)
    return produces_sample


def load_emb_data(emb_dict_file):
    word_dict = {}
    word_list = []
    item = 0
    with open(emb_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            word_dict[word] = item
            item += 1
            word_list.append(word)
    length = len(word_dict)
    print("Load embedding success! Num: %d" % length)
    return word_dict, length, word_list


def gen_id_data(emb_id_file, emb_data_file, vocab_dict):
    conjunction = ' '
    g = open(emb_id_file, 'w')
    f = open(emb_data_file, 'r')
    for data in f:
        id_data = [str(vocab_dict[i]) for i in data[:-1]]
        g.write(conjunction.join(id_data) + '\n')
    f.close()
    g.close()


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(200):  # data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def build_from_ids(vv, vocab_list):
    a = []
    for i in vv:
        a.append(vocab_list[i])
    return(' '.join(a))


def data_zoom(data, interval, alpha=1e-8):
    data = [alpha * (interval[1] - interval[0]) * (i - min(data)) / (max(data) - min(data)) for i in data]
    return data


def aliased_reward(aliased_prefixes, samples, rewards):
    aliased_rewards = data_zoom([i / MAX_SEQ_LENGTH for i in range(1, MAX_SEQ_LENGTH + 1)], [1e-20, 1])
    for i, sample in enumerate(samples):
        for aliased_prefix in aliased_prefixes:
            if aliased_prefix == sample[:len(aliased_prefix)]:
                rewards[i] = aliased_rewards[:len(aliased_prefix)].extend(rewards[i][len(aliased_prefix):])
                break
    return rewards


def create_negative_file_path(generator_num):
    negative_file_list = []
    for i in range(generator_num):
        negative_file = save_path + 'generator_' + str(i + 1) + '_sample.txt'
        negative_file_list.append(negative_file)
    return negative_file_list


def seed_classification(method_id=0, classifier=RFCBased(BATCH_SIZE)):
    if method_id == 0:
        print("Classification method: RFC Based")
        classifier = RFCBased(BATCH_SIZE)
    elif method_id == 1:
        print("Classification method: Entropy Clustering")
        classifier = EntropyClustering(BATCH_SIZE, k=6)
    elif method_id == 2:
        print("Classification method: IPv62Vec")
        classifier = IPv62Vec(BATCH_SIZE)
    elif method_id == -1:
        print("Classification method: None")
    return classifier


def main():

    # data pre-processing
    data_preprocessing = DataProcessing(emb_data_file, emb_dict_file, TOTAL_GENERATION)
    data_preprocessing.create_work_data(source_file, work_file)
    data_preprocessing.ip_split(source_file)
    data_preprocessing.gen_vocab()

    # load embedding info
    vocab_dict, vocab_size, vocab_list = load_emb_data(emb_dict_file)

    # seed classification
    positive_file_list = [emb_id_file]
    if CLASSIFICATION_METHOD != -1:
        classifier = seed_classification(CLASSIFICATION_METHOD)
        classifier.create_category()
        positive_file_list = classifier.gen_id_file(vocab_dict)
    generator_num = len(positive_file_list)

    # prepare data
    aliased_prefixes = []
    if ALIAS_DETECTION:
        prefix_loader = PrefixLoader(aliased_prefix_file)
        aliased_prefixes = prefix_loader.load_prefixes()

    gen_id_data(emb_id_file, emb_data_file, vocab_dict)
    pre_train_data_loaders = np.array([GenDataLoader(BATCH_SIZE, vocab_dict) for i in range(generator_num)])
    for i, positive_file in enumerate(positive_file_list):
        pre_train_data_loaders[i].create_batches([positive_file])

    gen_data_loaders = np.array([GenDataLoader(BATCH_SIZE, vocab_dict) for i in range(generator_num)])
    for i, positive_file in enumerate(positive_file_list):
        gen_data_loaders[i].create_batches([positive_file])

    dis_data_loader = DisDataLoader(BATCH_SIZE, vocab_dict, MAX_SEQ_LENGTH)

    # build model
    # num_emb, vocab_dict, batch_size, emb_dim, num_units, sequence_length

    generators = np.array([Generator(vocab_size, vocab_dict, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, MAX_SEQ_LENGTH, i)
                           for i in range(generator_num)])

    discriminator = Discriminator(sequence_length=MAX_SEQ_LENGTH, num_classes=generator_num + 1,
                                  vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    print('Generator-Data matching')
    for i in range(0, generator_num):
        print('Generator %s - Data %s' % (i + 1, positive_file_list[i]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    log = open(log_file, 'w')

    buffer = 'Start pre-training generator...'
    print(buffer)
    log.write(buffer + '\n')
    for i in range(generator_num):
        print('    Generator %s/%s' % (i + 1, generator_num))
        for epoch in range(50):  #120 # 150
            train_loss = pre_train_epoch(sess, generators[i], pre_train_data_loaders[i])
            if epoch % 5 == 0:
                generate_samples(sess, generators[i], 1, eval_file, vocab_list, if_log=True, epoch=epoch)
                print('    pre-train epoch ', epoch, 'train_loss ', train_loss)
                buffer = '    epoch:\t' + str(epoch) + '\tnll:\t' + str(train_loss) + '\n'
                log.write(buffer)

    buffer = 'Start pre-training discriminator...'
    print(buffer)
    log.write(buffer)
    negative_file_list = create_negative_file_path(generator_num)
    for filename in os.listdir(save_path):
        if 'generator' in filename:
            os.remove(save_path + filename)
    for _ in range(10):   # 10
        for i in range(generator_num):
            if CLASSIFICATION_METHOD == -1:
                generate_samples(sess, generators[i], int(TOTAL_GENERATION / BATCH_SIZE), negative_file_list[i],
                                 vocab_list)
            else:
                generate_samples(sess, generators[i], int(classifier.emb_data_num[i] / BATCH_SIZE), negative_file_list[i],
                                 vocab_list)
        dis_data_loader.load_train_data(positive_file_list, negative_file_list)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                }
                d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
        buffer = "    discriminator loss %f acc %f" % (d_loss, d_acc)
        print(buffer)
        log.write(buffer + '\n')

    print("Start Adversarial Training...")
    log.write('adversarial training...')
    rewards_loss_list = []
    for total_batch in range(1, TOTAL_BATCH + 1):
        # Train the generator
        for it in range(2):
            rewards_loss_list = []
            for i in range(generator_num):

                samples = generators[i].generate(sess)
                samples = produce_samples(samples)
                rewards = generators[i].get_reward(sess, samples, 16, discriminator)
                if ALIAS_DETECTION:
                    rewards = aliased_reward(aliased_prefixes, samples, rewards)
                a = str(samples[0])
                b = str(rewards[0])
                d = build_from_ids(samples[0], vocab_list)
                buffer = "%s\n%s\n%s\n%s\n\n" % (i + 1, d, a, b)
                print(buffer)
                log.write(buffer)
                rewards_loss = generators[i].update_with_rewards(sess, samples, rewards)


                # little1 good reward
                little1_samples = gen_data_loaders[i].next_batch()
                rewards = generators[i].get_reward(sess, little1_samples, 16, discriminator)
                if ALIAS_DETECTION:
                    rewards = aliased_reward(aliased_prefixes, samples, rewards)
                a = str(little1_samples[0])
                b = str(rewards[0])
                buffer = "%s\n%s\n\n" % (a, b)
                # print(buffer)
                log.write(buffer)
                rewards_loss = generators[i].update_with_rewards(sess, little1_samples, rewards)
                rewards_loss_list.append(rewards_loss)

        # Test
        if total_batch % 5 == 0:
            for i in range(generator_num):
                print('Generator %s/%s' % (i + 1, generator_num))
                generate_infer(sess, generators[i], total_batch, vocab_list, i + 1)
                buffer = 'reward-train epoch %s train loss %s' % (str(total_batch), str(rewards_loss_list[i]))
                print(buffer)
                log.write(buffer + '\n')
                generators[i].save_model(sess, model_path, str(i + 1))

        # Train the discriminator
        begin = True
        for _ in range(1):
            for i in range(generator_num):
                if CLASSIFICATION_METHOD == -1:
                    generate_samples(sess, generators[i], int(TOTAL_GENERATION / BATCH_SIZE), negative_file_list[i],
                                     vocab_list)
                else:
                    generate_samples(sess, generators[i], int(classifier.emb_data_num[i] / BATCH_SIZE),
                                     negative_file_list[i], vocab_list)
            dis_data_loader.load_train_data(positive_file_list, negative_file_list)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                    }
                    d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op],
                                                feed)
                    if total_batch % 5 == 0 and begin:
                        buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
                        print(buffer)
                        log.write(buffer)
                        begin = False
            discriminator.save_model(sess, model_path)

        # pretrain
        for _ in range(10):
            for i in range(generator_num):
                pre_train_epoch(sess, generators[i], pre_train_data_loaders[i])


if __name__ == '__main__':
    main()


