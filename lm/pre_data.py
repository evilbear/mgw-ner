import os

#Replace words with dictionary subscripts.
def conversion(vocab_words, path, save_path):
    data = open(path, 'r')
    save_data = open(save_path, 'w')
    for line in data:
        line = line.strip().split(' ')
        new_line = []
        for word in line:
            if word.isdigit():
                new_line.append(vocab_words["$NUM$"])
            elif word in vocab_words:
                new_line.append(vocab_words[word])
            else:
                new_line.append(vocab_words["$UNK$"])
        for i in new_line[:-1]:
            save_data.write(str(i)+' ')
        save_data.write(str(new_line[-1])+'\n')
    data.close()
    save_data.close()

#Convert the word of conll format annotated data into sentence form.
def unite(path, save_path):
    data = open(path, 'r')
    save_data = open(save_path, 'w')
    for line in data:
        line = line.strip()
        if len(line) != 0:
            save_data.write(line.split(' ')[0] + ' ')
        else:
            save_data.write('\n')
    data.close()
    save_data.close()

#Follow NER epochs for segmentation, in order to facilitate the use of LM learning vector sets for NER model.
#Parameter 20 from batch_size in ner model.
def split_file(file_name):
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/' + file_name + '/' + file_name + '.txt'
    temp_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/' + file_name + '/' + file_name + '_'
    data = open(path, 'r')
    line_num = 0 
    lineline = "" 
    for line in data:
        line_num += 1
        lineline += line.strip() + '\n'
        if line_num %20 ==0:
            save_path = temp_path + str(int(line_num/20)-1)+'.txt'
            save_file = open(save_path, 'w')
            save_file.write(lineline.strip('\n'))
            save_file.close()
            lineline = ""
    if len(lineline) != 0:
        save_path = temp_path + str(int(line_num/20))+'.txt'
        save_file = open(save_path, 'w')
        save_file.write(lineline.strip('\n'))
        save_file.close()
    data.close()

def con_data(vocab_words):
    #conversion lm data
    train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train.txt'
    con_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/lm_train.txt'
    conversion(vocab_words, train_path, con_train_path)

    dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev.txt'
    con_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/lm_dev.txt'
    conversion(vocab_words, dev_path, con_dev_path)

    #conversion ner data
    ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/train.txt'
    temp_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train/train_.txt'
    unite(ner_train_path, temp_train_path)
    con_ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train/train.txt'
    conversion(vocab_words, temp_train_path, con_ner_train_path)

    ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/dev.txt'
    temp_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev/dev_.txt'
    unite(ner_dev_path, temp_dev_path)
    con_ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev/dev.txt'
    conversion(vocab_words, temp_dev_path, con_ner_dev_path)

    ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/test.txt'
    temp_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test/test_.txt'
    unite(ner_test_path, temp_test_path)
    con_ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test/test.txt'
    conversion(vocab_words, temp_test_path, con_ner_test_path)

    #split ner data
    split_file('train')
    split_file('dev')
    split_file('test')

