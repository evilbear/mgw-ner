# Attention-based BLSTM-CRF Architecture for Mongolian Named Entity Recognition

系统：ubuntu 16.04 server
语言：python3
版本：Anaconda3-5.1.0
框架：Tensorflow-gpu 1.7.0

1.使用Glove工具对预先准备好的无标注数据进行训练，获取预训练的词嵌入。
2.使用无标注数据构建基于两层LSTM和Softmax的神经语言模型；考虑到单词上文和下文的信息都有效，训练正向和反向两个LM，独立训练参数无关，区别是反向对输入进行翻转。
训练：python lm/lm_main.py 
     python lm/lm_main.py --use_model=bw_model
3.LM中通过LSTM学习的向量包含单词的语义和句法角色，我们进行截断输出，不进行Softmax，直接把这个LM向量作为外部信息传入NER模型。在1中预训练好了LM，我们按照NER模型参数的批次对标注数据切分后进行训练，准备对应单词的LM向量。
获取：python lm/lm_main.py --mode=test
     python lm/lm_main.py --use_model=bw_model --mode=test
4.构建基于BLSTM和CRF的命名实体识别模型；先对单词进行字符级BLSTM学习，获取字符级向量，并和预训练的词向量进行拼接放入词级BLSTM学习BLSTM向量；和3中准备好的LM向量使用注意力机制进行结合，然后放入CRF层进行联合解码，最终使用Viterbi算法找出最可能的标签序列。
训练：python ner/ner_main.py --output_path=/result/lms_att/
测试：python ner/ner_main.py --output_path=/result/lms_att/ --mode=test
5.注意力机制：通过权重矩阵动态决定两个输入对标签预测的贡献程度，使用加权和；z是权重矩阵，Wz是参数，Hk和Mk分别是输入的LSTM向量和LM向量，Yk是注意力机制层的输出，公式如下：![image](https://github.com/evilbear/mgw-ner/blob/master/src/picture/1.png)
![image](https://github.com/evilbear/mgw-ner/blob/master/src/picture/2.png)
![image](https://github.com/evilbear/mgw-ner/blob/master/src/picture/3.png)
