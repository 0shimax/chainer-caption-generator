import chainer
import chainer.functions as F
import chainer.links as L


class Attention(chainer.Chain):
    """
    input to lstm is output of attention(fb_t)
    hidden state in lstm is f_{t-1}
    in case image, input to align_source_dim is feature map reduced channel to one.
    then, align_score_dim is (batch_size, n_hid)
    hidden_state is (batch_size, hidden_state_dim)
    hidden_state_dim is (reviewer=rnn_size, decoder=voc_size)
    """
    def __init__(self, align_source_dim, hidden_state_dim):
        super().__init__(
            up_dim_to_source= \
                    L.Linear(hidden_state_dim, align_source_dim),  # (batch_size, rnn_size) -> (batch_size, n_hid)
            reduce_dim_to_hidden_state_dim= \
                    L.Linear(align_source_dim, hidden_state_dim),  # (batch_size, n_hid) ->(batch_size, rnn_size)
        )

    def __call__(self, align_source, pre_hidden_state):
        dot = self.up_dim_to_source(pre_hidden_state)
        weight = F.softmax(dot)
        att_res = self.reduce_dim_to_hidden_state_dim(weight*align_source)
        return att_res
