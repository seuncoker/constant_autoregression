import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import math
import copy

from constant_autoregression.util import Printer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




######################################################################
######################################################################
######################################################################



def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([module.to(device) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    #attn_shape = (1,1, size, size)
    attn_shape = (1,size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    ).to(device)
    return subsequent_mask == 0




######################################################################
######################################################################
######################################################################



class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model):
        super(Generator, self).__init__()




        self.x_function = None
        self.amplitude = None
        self.phase = None

        self.input_func = None

    def forward(self, x, modes_out, res):

        r = x[:,:,:,0:1,:]
        i = x[:,:,:,1:2,:]

        x_ft = torch.complex(r, i).to(device)

        x_ft = x_ft.permute(0,1,2,4,3)
        print("x_ft -->", x_ft.shape)

        a =  torch.abs(x_ft)
        p = torch.angle(x_ft)

        self.amplitude =  a[:,:,:,:modes_out,:]
        self.phase = p[:,:,:,:modes_out,:]

        print("amplitude -->", self.amplitude.shape)
        print("phase -->", self.phase.shape)


        domain_length = torch.tensor([1]).to(device)

        x_res = torch.linspace(0,1,res).to(device)
        frequencies =  torch.fft.fftfreq(res, d= x_res[1] - x_res[0])[:modes_out].unsqueeze(dim=-1)
        res_original = res
        resolution = torch.linspace(0,1,res_original).to(device)

        input_vec = (self.amplitude.unsqueeze(-1)) * torch.cos( (2*torch.pi*(frequencies*resolution)/domain_length).unsqueeze(0).unsqueeze(2) + self.phase.unsqueeze(-1)  )  ###
        print("input_vec ->", input_vec.shape)
        self.input_func = input_vec.permute(0,1,5,2,3,4)
        x_out_2 = self.input_func.sum(-1).sum(-1)
        print("x_out_2 -->", x_out_2.shape)
        return x_out_2




######################################################################
######################################################################
######################################################################




class Lifting(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model):
        super(Lifting, self).__init__()
        self.lifting = nn.Linear(1,d_model)
        self.x = None

    def forward(self, x):
        #print("Lifting ...")
        #print("x -->", x.shape)
        self.x = self.lifting(x.unsqueeze(-1))
        #self.x = x.unsqueeze(-1)
        #print("x_lift -->", x.shape)
        return self.x


class Projecting(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model):
        super(Projecting, self).__init__()
        self.project = nn.Linear(d_model,1)
        self.x = None

    def forward(self, x):
        #print("Projecting...")
        #print(x.shape)
        self.x = self.project(x)
        #print(x.shape)
        #self.x = x
        return self.x.squeeze(-1)



######################################################################
######################################################################
######################################################################




class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # self.w_k_1 = nn.Linear(d_model, d_ff)
        # self.w_k_2 = nn.Linear(d_ff, d_model)

        # self.w_q_1 = nn.Linear(d_model, d_ff)
        # self.w_q_2 = nn.Linear(d_ff, d_model)

        # self.w_v_1_a = nn.Linear(d_model, d_ff)
        # self.w_v_2_a = nn.Linear(d_ff, d_model)

        # self.w_v_1_p = nn.Linear(d_model, d_ff)
        # self.w_v_2_p = nn.Linear(d_ff, d_model)
        d_model = 100
        self.w_v_1_a = nn.Linear(d_model, d_model)
        self.w_v_2_a = nn.Linear(d_model, d_model)

        self.w_v_1_p = nn.Linear(d_model, d_model)
        self.w_v_2_p = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        #self.act = nn.GELU()
        self.act_a = nn.ReLU()
        self.act_p = nn.Tanh()


        self.q = None
        self.k = None
        self.v = None

        self.q_ffn = None
        self.k_ffn = None
        self.v_ffn = None
        #self.act_p = nn.Tanh()

    def forward(self, q, k, v):
        #print("Doing Feedforward NN...")
        self.q = q
        self.k = k
        self.v = v
        #
        #print("v-->",v.shape)
        #x = x.permute(0,1,2,4,3)
        #x = x
        # x_a = self.w_2(self.act(self.w_1(x[:,:,0,...])))
        # x_p = self.w_2_p(self.act(self.w_1_p(x[:,:,1,...])))
        # x = torch.cat((x_a.unsqueeze(2),x_p.unsqueeze(2)), dim=2)

        # self.q_ffn = self.w_q_2(self.act(self.w_q_1(q)))
        # self.k_ffn = self.w_k_2(self.act(self.w_k_1(k)))
        v_a = v[:,:,:,0:1]
        v_p = v[:,:,:,1:2]
        v_ffn_a = self.w_v_2_a(self.act_a(self.w_v_1_a(v_a)))
        v_ffn_p = self.w_v_2_p(self.act_p(self.w_v_1_p(v_p)))
        self.v_ffn = torch.cat((v_ffn_a,v_ffn_p), dim=3)
        #print("self.vffn-->",self.v_ffn.shape)

        #x = x.permute(0,1,2,4,3)
        #print("x -->", x.shape)

        #x = self.w_2(self.act(self.(x.permute(0,1,2,4,3))))
        return q, k, self.v_ffn





######################################################################
######################################################################
######################################################################





class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.eps = eps

        self.x_norm = None
        self.x = None

    def forward(self, x):
        self.x = x.clone()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        self.x_norm = (x - mean)/ (std + self.eps)

        return self.x_norm




######################################################################
######################################################################
######################################################################

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    original_query_shape = query.shape


    query = query.permute(0,4,1,2,3,5).unsqueeze(4).unsqueeze(5)
    key = key.permute(0,4,1,2,3,5).unsqueeze(2).unsqueeze(3)
    value = value.permute(0,4,1,2,3,5).unsqueeze(2).unsqueeze(3)

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.permute(0,1,3,5,6,7,2,4)
        #print("scores -->", scores.shape)
        scores = scores.masked_fill(mask == 0, -1e9).to(device)
        #print("scores -->", scores.shape)
        scores = scores.permute(0,1,6,2,7,3,4,5)

    #print("scores -->  ", scores.shape)
    scores_1 = scores.permute(0,1,3,5,2,6,4,7)
    scores_2 = scores_1.reshape(scores_1.shape[0],scores_1.shape[1], scores_1.shape[2], scores_1.shape[3], scores_1.shape[4]*scores_1.shape[5], scores_1.shape[6]*scores_1.shape[7])

    p_attn = scores_2.softmax(-1)
    p_attn = p_attn.reshape(scores_1.shape[0],scores_1.shape[1], scores_1.shape[2], scores_1.shape[3], scores_1.shape[4], scores_1.shape[5], scores_1.shape[6], scores_1.shape[7])
    p_attn_ = p_attn.permute(0,1,4,2,6,3,5,7)

    p_attn = p_attn_.mean(4, keepdim=True).mean(5, keepdim=True)#.to(device)

    result = torch.matmul(p_attn, value)#.to(device)
    result = result.mean(4).mean(4)#.to(device)

    result = result.permute(0,2,3,4,1,5)


    return result, p_attn_








######################################################################
######################################################################
######################################################################





class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, modes_in, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        #self.d_k = d_model // h
        #h = 5
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)#.to(device)
        self.attn = None

        self.x = None

        self.query = None
        self.key = None
        self.value = None
        self.modes_in = modes_in

        self.dropout = nn.Dropout(p=dropout)#.to(device)

    def forward(self, q,k,v, mask=None):
        #print("Doing attention.....")
        #print("query, key, value", q.shape, k.shape, v.shape)
        query, key, value = q.permute(0,1,3,4,2), k.permute(0,1,3,4,2), v.permute(0,1,3,4,2)
        #print("query, key, value", query.shape, key.shape, value.shape)

        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(0)#.to(device)

        #print("query -->", query)
        nbatches = query.size(0)
        ntimes = query.size(1)
        res = query.size(2)
        q_nfunctions = query.size(3)
        kv_nfunctions = key.size(3)
        dim = query.size(4)


        # self.query, self.key, self.value = [
        #     lin(x).view(nbatches, -1, res,  nfunctions, self.h, self.d_k)
        #     for lin, x in zip(self.linears, (query, key, value))
        # ]

        self.query = self.linears[0](query).view(nbatches, -1, res,  q_nfunctions, self.h, self.d_k)
        self.key = self.linears[1](key).view(nbatches, -1, res,  kv_nfunctions, self.h, self.d_k)
        self.value = self.linears[2](value).view(nbatches, -1, res,  kv_nfunctions, self.h, self.d_k)

        self.x, self.attn = attention(
            self.query, self.key, self.value, mask=mask, dropout=self.dropout
        )


        self.x = (
            self.x
            .contiguous()
            .view(nbatches, ntimes, res, q_nfunctions, self.h * self.d_k)
        )

        self.query = (
            self.query
            .contiguous()
            .view(nbatches, ntimes, res, q_nfunctions, self.h * self.d_k)
        )

        self.key = (
            self.key
            .contiguous()
            .view(nbatches, ntimes, res, kv_nfunctions, self.h * self.d_k)
        )

        #print("self.query, self.key,  self.x -->", self.query.shape, self.key.shape,  self.x.shape)


        self.query, self.key,  self.x = self.query.permute(0,1,4,2,3), self.key.permute(0,1,4,2,3),  self.x.permute(0,1,4,2,3)
        #print("query, key, x -->", self.query.shape, self.key.shape,  self.x.shape)
        self.x = self.x[...,:self.modes_in]
        #print("query, key, x -->", self.query.shape, self.key.shape,  self.x.shape)

        return self.query, self.key,  self.x








######################################################################
######################################################################
######################################################################



class Embedding(nn.Module):
    def __init__(self, d_model):
        super(Embedding, self).__init__()

        #self.frequency_embedding = nn.Embedding(vocab, d_model)

        #self.lifting = nn.Linear(1,d_model)

        self.src_function = None
        self.src_frequency = None
        self.src_amplitude = None
        self.src_phase =  None

        self.amplitude = None
        self.phase =  None

        self.input_embedding = None
        self.frequency_embedding = None
        self.time_embedding_kv = None
        self.time_embedding_q = None

        self.q = None
        self.kv = None


        self.d_model = d_model
        # constant =  10000
        # sequence_length = 1000

        # postional_embed = torch.arange(sequence_length).unsqueeze(1)#.to(device)
        # self.div_term = postional_embed / torch.tensor([math.pow(constant, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)])#.to(device)
        # self.postional_embedding = torch.zeros(sequence_length, self.d_model)#.to(device)
        # self.postional_embedding[:, 0::2] = torch.sin(self.div_term[:, 0::2])#.to(device)
        # self.postional_embedding[:, 1::2] = torch.cos(self.div_term[:, 1::2])#.to(device)



        self.frequency_embed = nn.Linear(1, 1)
        self.time_embedding = nn.Linear(1, 1)
        self.x = None
        # self.frequency_embed_time = self.frequency(torch.arange(self.vocab))


    def forward(self, x, kv_time, q_time, modes_in):
        #print("\n")
        #print("Doing Embedding ...")
        #print("x -->", x.shape)
        self.x = x
        src_function, src_frequency , src_amplitude, src_phase = self.frequency_encoding(x)
        #print("src_function, src_frequency , src_amplitude, src_phase ->", src_function.shape, src_frequency.shape , src_amplitude.shape, src_phase.shape)
        self.src_function = src_function#[:, :, :, :modes_in, :]
        self.src_frequency = src_frequency#[:modes_in,:]
        self.src_amplitude = src_amplitude#[..., :modes_in]
        self.src_phase = src_phase#[..., :modes_in]



        input_embed  = torch.cat((self.src_amplitude, self.src_phase), dim = 3)
        #print("input_embed -->",input_embed.shape)
        self.input_embedding = input_embed

        frequency_embed = self.frequency_embed(self.src_frequency.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0))
        self.frequency_embedding = frequency_embed.squeeze(-1)
        #print( "frequency_embed -->",  self.frequency_embedding.shape )



        #print("kv_time, q_time -->", kv_time.shape, q_time.shape)
        #print("kv_time ->", kv_time[:3,:])
        #print("q_time ->", q_time[:3,:])
        time_embed_kv = self.time_embedding(kv_time.unsqueeze(-1).float())
        time_embed_q = self.time_embedding(q_time.unsqueeze(-1).float())
        #print("time_embed_kv time_embed_q -->", time_embed_kv.shape, time_embed_q.shape)


        self.time_embedding_kv = time_embed_kv.unsqueeze(-1).unsqueeze(-1)#.unsqueeze(2).unsqueeze(3).repeat(1,1,input_embed.shape[2], modes_in, 1)
        self.time_embedding_q = time_embed_q.unsqueeze(-1).unsqueeze(-1)#.unsqueeze(2).unsqueeze(3).repeat(1,1,input_embed.shape[2], modes_in, 1)
        #print("time_embedding_kv -->", self.time_embedding_kv.shape)
        #print("time_embedding_q -->", self.time_embedding_q.shape)


        #print()
        # self.kv = self.input_embedding*(self.frequency_embedding + self.time_embedding_kv)
        # self.q = self.input_embedding*(self.frequency_embedding + self.time_embedding_q)

        # self.kv = self.input_embedding[..., :modes_in]*(self.frequency_embedding[..., :modes_in] + self.time_embedding_kv)
        # self.q = self.input_embedding*( self.frequency_embedding + self.time_embedding_q)

        self.kv = self.input_embedding[..., :modes_in] * ( self.time_embedding_kv)
        self.q = self.input_embedding * ( self.time_embedding_q)
        #print("x -->", self.kv.shape)

        # self.kv = self.input_embedding[..., :modes_in]#[...,:modes_in]# + self.frequency_embedding + self.time_embedding_kv
        # self.q = self.input_embedding# + self.frequency_embedding + self.time_embedding_q
        #print("q, kv -->", self.q.shape, self.kv.shape)
        return self.q, self.kv, self.kv


    def frequency_encoding(self, input):
        #print( input.shape)
        input = input.permute(0,1,3,2)
        input_shape = input.shape
        #print( input.shape)
        batch_size = input_shape[0]
        time_size = input_shape[1]
        dim_size = input_shape[2]
        res = input_shape[3]

        domain_length = torch.tensor([1]).to(device)
        x_res = torch.tensor(torch.linspace(0, 1, res), dtype=torch.float).to(device)

        #input_freq_comp = torch.zeros((batch_size,res,res))#.to(device)
        #print("input -->", input.shape)
        # print("input.squeeze(dim=-1) -->", input.squeeze(dim=-1).shape)
        input_fft = torch.fft.fft(input)[...,:res//2].to(device)
        frequencies =  torch.fft.fftfreq(res, d= x_res[1] - x_res[0])[...,:res//2].unsqueeze(dim=-1).to(device)

        amplitude = input_fft.real.unsqueeze(dim=-1)
        phase = input_fft.imag.unsqueeze(dim=-1)

        amp = torch.abs(input_fft)/float(res//2)#.to(device)
        p = torch.angle(input_fft)#.to(device)

        # print("input_fft -->", input_fft.shape)
        # print("frequencies -->", frequencies.shape)
        # print("amplitude -->", amplitude.shape)
        # print("phase -->", phase.shape)

        x = 2*torch.pi*frequencies*x_res/domain_length
        #print(x.shape)
        x_function  = (amp.unsqueeze(-1)) * torch.cos( (2*torch.pi*(frequencies*x_res)/domain_length) + p.unsqueeze(-1)  )  ###
        #print("x_function -->", x_function.shape)
        # x_function = x_function.permute(0,1,5,2,3,4)
        # print("x_function -->", x_function.shape)





        # x_res = torch.linspace(0,1,self.res)
        # frequencies =  torch.fft.fftfreq(self.res, d= x_res[1] - x_res[0])[:self.modes_out].unsqueeze(dim=-1)
        # res_original = x.shape[-1]
        # resolution = torch.linspace(0,1,res_original)

        # input_vec = (amplitude.unsqueeze(-1)) * torch.cos( (2*torch.pi*(frequencies*resolution)/domain_length).unsqueeze(0).unsqueeze(2) + phase.unsqueeze(-1)  )  ###
        # print("input_vec ->", input_vec.shape)
        # self.input_func = input_vec.permute(0,1,5,2,3,4)



        #print(input_freq_comp.shape)
        return x_function, frequencies, amplitude.permute(0,1,2,4,3), phase.permute(0,1,2,4,3)






######################################################################
######################################################################
######################################################################

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)
        self.q = None
        self.k = None
        self.v = None

        self.q_norm = None
        self.k_norm = None
        self.v_norm = None

        self.res_q = None
        self.res_k = None
        self.res_v = None

        self.q_res = None
        self.k_res = None
        self.v_res = None

        #self.act = nn.Tanh()
        self.act = nn.GELU()



    def forward(self, q, k, v, sublayer, connection):
        "Apply residual connection to any sublayer with the same size."
        #print("Doing Sublayer connections: Norm -> dropout -> residual ->  x +  residual")
        self.q = q
        self.k = k
        self.v = v

        # self.q_norm = self.norm(self.q)
        # self.k_norm = self.norm(self.k)
        # self.v_norm = self.norm(self.v)

        #print("Spectral Convolution ...")
        #self.res_q, self.res_k, self.res_v = sublayer(self.q, self.k, self.v)

        # if activation:
        #   #print("Activation .....")
        #   self.q_res = self.res_q + self.act(self.res_q)
        #   self.k_res = self.res_k + self.act(self.res_k)
        #   self.v_res = self.res_v + self.act(self.res_v)

        if connection == "_activation":
          #print("Activation .....")
          self.res_v = sublayer(self.v)

          self.v_res = self.act(self.res_v)


        elif connection == "_":
          #print(" NO Activation .....")
          self.res_v = sublayer(self.v)
          self.v_res = self.res_v


        elif connection == "_residual":
          self.res_q, self.res_k, self.res_v = sublayer(self.q, self.k, self.v)
          #print("Residual")
          self.q_res = self.q + self.res_q
          self.k_res = self.k + self.res_k
          self.v_res = self.v + self.res_v

        elif connection == "norm_residual":
          #print("Norm residual")
          self.q_norm = self.norm(self.q)
          self.k_norm = self.norm(self.k)
          self.v_norm = self.norm(self.v)

          self.res_q, self.res_k, self.res_v = sublayer(self.q_norm, self.k_norm, self.v_norm)
          #print(self.res_q.shape, self.q.shape)
          self.q_res = self.q + self.res_q
          self.k_res = self.k + self.res_k
          self.v_res = self.v + self.res_v

        elif connection == "activation_residual":

          self.v_act = self.act(self.v)

          self.res_v = sublayer(self.v_act)

          self.v_res = self.v + self.res_v

        elif connection == "_activationresidual":
          #print("activation residual")
          # self.q_act = self.act(self.q)
          # self.k_act = self.act(self.k)
          # self.v_act = self.act(self.v)

          self.res_q, self.res_k, self.res_v = sublayer(self.q, self.k, self.v)

          self.q_res = self.q + self.act(self.res_q)
          self.k_res = self.k + self.act(self.res_k)
          self.v_res = self.v + self.act(self.res_v)

        elif connection == "_residualactivation":
          #print("activation residual")
          # self.q_act = self.act(self.q)
          # self.k_act = self.act(self.k)
          # self.v_act = self.act(self.v)

          self.res_q, self.res_k, self.res_v = sublayer(self.q, self.k, self.v)

          self.q_res = self.act(self.q + self.res_q)
          self.k_res = self.act(self.k + self.res_k)
          self.v_res = self.act(self.v + self.res_v)

        else:
          raise TypeError("Incorrect argument.......")



        #print("self.q_res, self.k_res, self.v_res -->", self.q_res.shape, self.k_res.shape, self.v_res.shape)
        return self.q_res, self.k_res, self.v_res













######################################################################
######################################################################
######################################################################




class SpectralConv1d_wave(nn.Module):
    def __init__(self, modes_in, modes_out, time_seq, hidden_dimension):
        super(SpectralConv1d_wave, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.modes_in = modes_in
        self.modes_out = modes_out
        self.time_seq = time_seq
        self.d_model = hidden_dimension

        self.res = 200

        self.scale = (1 / (self.modes_in*self.modes_out))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.time_seq, 1, 1,  self.modes_out,  self.modes_in, dtype=torch.cfloat)).to(device)
        #self.weights1 = nn.Parameter(self.scale * torch.rand(self.time_seq, 1, 1, self.modes_out, self.modes_in, dtype=torch.cfloat))
        self.act_p = nn.Tanh()
        self.act_a = nn.ReLU()

        self.input_func = None
        self.x = None
        self.x_ft_1 = None
        self.x_ft_2 = None
        self.v = None

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        #print("input, weight -->", input.shape, weights.shape)
        return torch.einsum("bthix,thiox->bthox", input, weights)


    def forward(self, v):
        #print("Doing spectral conv...")
        batchsize = v.shape[0]
        #print("v -->", v.shape)
        self.v = v
        r = v[:,:,:,0:1,:]#/float(self.res//2)
        i = v[:,:,:,1:2,:]
        #print("a , p ", a.shape, p.shape)

        x_ft = torch.complex(r, i).to(device)


        self.x_ft_2 = x_ft
        #print("x_ft -->", x_ft.shape)
        #print("modes_out,  modes_in -->", self.modes_out, self.modes_in)
        out_ft = torch.zeros(batchsize, self.time_seq, self.d_model, self.modes_out, self.modes_in, dtype=torch.cfloat).to(device)
        #print("out_ft -->", out_ft.shape)
        out_ft[..., :self.modes_in] = self.compl_mul1d(x_ft[:, :, :, :, :self.modes_in], self.weights1)
        # print("out_ft -->", out_ft.shape)
        # print("v -->", v.shape)
        #out_ft += x_ft

        #print("out_ft -->", out_ft.shape)

        # out_real = out_ft.real.sum(-1, keepdim=True)
        # out_imag = out_ft.imag.sum(-1, keepdim=True)

        # print("out_real, out_imag -->", out_real.shape, out_imag.shape)
        # v_out = torch.cat((out_real, out_imag), dim = -1)
        # print("v_out -->", v_out.shape)
        # v_out = v_out.permute(0,1,2,4,3)
        # print("v_out -->", v_out.shape)



        #Reconstruct the plane waves
        #amplitude =  torch.abs(out_ft)#/float(self.res//2)
        amplitude =  self.act_a(torch.abs(out_ft))#/float(self.res//2)
        phase = torch.angle(out_ft)
        #phase = 2*torch.pi*self.act_a(torch.angle(out_ft))
        #print("amplitude -->", amplitude.shape)
        #print("phase -->", phase.shape)

        domain_length = torch.tensor([1]).to(device)

        x_res = torch.linspace(0,1,self.res).to(device)
        frequencies =  torch.fft.fftfreq(self.res, d= x_res[1] - x_res[0])[:self.modes_out].unsqueeze(dim=-1).to(device)
        res_original = self.res
        resolution = torch.linspace(0,1,res_original).to(device)

        #qq = self.freq_to_space(amp, freq, res, l, p)
        input_vec = (amplitude.unsqueeze(-1)) * torch.cos( (2*torch.pi*(frequencies*resolution)/domain_length).unsqueeze(0).unsqueeze(2) + phase.unsqueeze(-1)  )  ###
        #print("input_vec ->", input_vec.shape)

        self.input_func = input_vec.permute(0,1,5,2,3,4)
        x_out_2 = self.input_func.sum(-1).sum(-1)
        #print("x_out_2 -->", x_out_2.shape)
        # print("q, k, x -->", q.shape, k.shape, x_out_2.shape)
        # print("\n")
        return x_out_2












######################################################################
######################################################################
######################################################################
class EncoderDecoder_wave(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, lifting, projecting):
        super(EncoderDecoder_wave, self).__init__()
        #self.embedding = embedding
        self.encoder = encoder
        #self.generator = generator
        self.lifting = lifting
        self.projecting = projecting
        self.q = None
        self.k = None
        self.v = None
        self.tgt_embed = None

        self.src = None
        self.tgt = None
        self.memory = None


    def forward(self, src, tgt, src_mask, tgt_mask, src_time, tgt_time, modes_in, modes_out):
        "Take in and process masked src and target sequences."
        src = self.lifting(src)
        encode = self.encode(src, tgt_mask, src_time, tgt_time, modes_in)
        self.memory = encode

        #tgt_out = self.generator(encode, modes_out, src.shape[-1])

        tgt_out = self.projecting(encode)

        return tgt_out


    def encode(self, src, src_mask, src_time, tgt_time, modes_in):
        #print("Starting Encoding")
        self.src = src
        #self.q, self.k, self.v = self.src, self.src, self.src
        #self.q, self.k, self.v = self.embedding(src, src_time, tgt_time, modes_in)
        return self.encoder(self.src, src_mask, src_time, tgt_time, modes_in)





######################################################################
######################################################################
######################################################################

class Encoder_wave(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, activation, embedding_s_to_f):
        super(Encoder_wave, self).__init__()
        self.layers = clones(layer, N)
        self.activation = activation
        self.embedding_s_to_f = embedding_s_to_f
        #self.embedding_f_to_s = embedding_f_to_s
        #self.norm = LayerNorm()

    def forward(self, x, mask, src_time, tgt_time, modes_in):
        "Pass the input (and mask) through each layer in turn."
        #print("STARTING ENCODER ....")
        count_en = 0
        q, k, v = x, x, x
        for i,layer in enumerate(self.layers):
            #print("Start Encoder: ", count_en)
            q, k, v = self.embedding_s_to_f(v, src_time, tgt_time, modes_in)
            v = layer(q, k, v, mask, self.activation[i])
            #q, k, v = self.embedding_f_to_s(q, k, v, modes_out)
            #print("Done Encoder: ", count_en)
            count_en += 1
        return v #+ self.norm(v)



######################################################################
######################################################################
######################################################################

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, self_attn, feed_forward, spectral_conv, dropout ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        #self.feed_forward = feed_forward
        self.spectral_conv = spectral_conv 
        self.sublayer = clones(SublayerConnection(dropout), 2)

    def forward(self, q, k, v, mask, connection_type):
        "Follow Figure 1 (left) for connections."
        #print("Start 1st SUB CONNECT")
        #x = self.embedding(x, time)
        #print("q, k, v  ", q.shape, k.shape, v.shape)
        q, k, v = self.sublayer[0](q, k, v, lambda q, k, v: self.self_attn(q, k, v, mask), connection_type[0])
        #print("q, k, v  ", q.shape, k.shape, v.shape)

        #q, k, v = self.embedding(x, src_time, tgt_time, modes_in)
        #_, _, v = self.sublayer[1](q, k, v, self.feed_forward, connection_type[1])
        #print("q, k, v  ", q.shape, k.shape, v.shape)

        _, _, v = self.sublayer[1](q, k, v, self.spectral_conv, connection_type[1] )
        #print("q, k, v  ", q.shape, k.shape, v.shape)

        #print("\n")
        #
        #print("Done 1st SUB CONNECT")

        #q, k, v = self.sublayer[1](q, k, v, self.feed_forward)
        #print("Done 2nd SUB CONNECT")
        return v