import sys
sys.path.append('./src/common/linker')
sys.path.append('./src/caption_generator')
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from attention import Attention
from mod_lstm import ModLSTM


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
        )

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class Encoder(chainer.Chain):
    def __init__(self, in_ch, n_unit):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            fire5=Fire(256, 32, 128, 128),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192),
            fire8=Fire(384, 64, 256, 256),  # (n_batch, 512, 44, 44), 44*44=1936
            conv9=L.Convolution2D(512*21, 4096, 1, pad=0),  # (n_batch, 1024)
            conv10=L.Convolution2D(4096, n_unit, 1, pad=0),
        )
        self.train = True

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        self.sorce_hidden_state = self.fire8(h)  # H in the paper.

        h = F.spatial_pyramid_pooling_2d( \
            self.sorce_hidden_state, 3, F.MaxPooling2D)
        h = F.elu(self.conv9(h))
        h = F.dropout(h, ratio=0.5, train=self.train)
        self.context_vec = self.conv10(h)


class Decoder(chainer.Chain):
    def __init__(self, align_source_size, n_unit, voc_size):
        super().__init__(
            attention = Attention(align_source_size, n_unit),
            dec_lstm = ModLSTM(n_unit, n_unit),
            l_out = L.Linear(n_unit, voc_size),
        )
        self.align_source = None
        self.pre_hidden_state = None
        self.train = True
        self.sentence_ids = []

    def clear(self):
        self.dec_lstm.reset_state()
        self.sentence_ids = []
        self.align_source = None
        self.pre_hidden_state = None

    def __call__(self, t):
        att = self.attention(self.align_source, self.pre_hidden_state)
        h_state = self.dec_lstm(att)
        y = self.l_out(F.dropout(h_state, train=self.train))  # don't forget to change drop out into non train mode.
        self.sentence_ids.append(F.argmax(y, axis=1).data)
        self.pre_hidden_state = h_state
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy,


class AttentionCaptionNetMod(chainer.Chain):
    def __init__(self, n_class, in_ch, voc_size=100, n_view_steps=8,
                                        n_unit=512, align_source_size=44*44):
        super().__init__(
            encoder=Encoder(in_ch, n_unit),
            reduce_feat_dim=L.Convolution2D(512, 1, 1, pad=0),
            decoder=Decoder(align_source_size, n_unit, voc_size),
        )
        self.train = True
        self.n_class = n_class
        self.active_learn = False
        self.align_source_size = align_source_size
        self.n_unit = n_unit
        self.voc_size = voc_size

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.decoder.clear()
        self.decoder.train = self.train
        self.encoder.train = self.train
        self.sentence_ids = None

    def forward_decoder(self, align_source, tokens):
        '''
        tokens: [[word11,word12,...,EOS],...,[word n1, word n2...,EOS]] as words in a text
        '''
        # extract words array(ex [w11,w21]) from words matrix(ex [[w11,w12,...],[w21,w22...]])
        # There are as many texts as there are batch size.

        batch_size, voc_len = tokens.data.shape
        self.decoder.align_source = align_source

        # pre_hidden_state = self.xp.zeros((batch_size, self.n_unit))
        # self.decoder.pre_hidden_state = Variable( \
        #     pre_hidden_state.astype(self.xp.float32), volatile='auto')
        self.decoder.pre_hidden_state = self.encoder.context_vec

        first_word = F.get_item(tokens, [range(batch_size), 0])
        self.loss, self.accuracy = self.decoder(first_word)

        neg_one_mat = Variable(self.xp.ones_like( \
            self.decoder.pre_hidden_state.data, self.xp.float32), volatile='auto')
        neg_one_mat *= -1

        # cur_word is current word in each sentence as 1d array.
        # next_word is next word in each sentence as 1d array.
        for i_word in range(1, voc_len):
            next_word = F.get_item(tokens, [range(batch_size), i_word])
            broaded_next_word = F.broadcast_to( \
                F.expand_dims(next_word, axis=1), neg_one_mat.data.shape)
            self.decoder.pre_hidden_state = F.where( \
                broaded_next_word.data==-1, neg_one_mat, self.decoder.pre_hidden_state)
            loss, acc = self.decoder(next_word)
            self.loss += loss
            self.accuracy += acc
        self.accuracy /= voc_len

    def __call__(self, x, t, tokens):
        self.clear()
        x.volatile = not self.train
        self.encoder(x)

        align_source = self.reduce_feat_dim(self.encoder.sorce_hidden_state)
        batch_size = len(align_source.data)
        align_source = F.reshape(align_source, (batch_size, -1))
        self.forward_decoder(align_source, tokens)

        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        self.sentence_ids = self.xp.array(self.decoder.sentence_ids)
        return self.loss
