import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia
from geofree.modules.warp.midas import Midas
from ldm.util import instantiate_from_config
from ldm.modules.x_transformer import (
    Encoder,
    TransformerWrapper,
)  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

from torch.distributions import Uniform


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        # 이게 key값, class임베딩 수행
        # camera info conditioning도 여기서 수행
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c  # b, 1, 512


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# condition 수행부분(카메라 파라미터)
# class CodeBookEmbedder(nn.Module):
#     def __init__(
#         self,
#         new_stage_config,
#         emb_stage_config,
#         emb_stage_key="camera",
#     ):
#         super().__init__()
#         self.init_first_stage_from_ckpt(new_stage_config)
#         self.init_emb_stage_from_ckpt(emb_stage_config)
#         self.emb_stage_key = emb_stage_key

#     def init_first_stage_from_ckpt(self, config):
#         model = instantiate_from_config(config)
#         self.first_stage_model = model.eval()
#         # self.first_stage_model= model.train()
#         self.first_stage_model.train = disabled_train

#     def init_emb_stage_from_ckpt(self, config):
#         if config is None:
#             self.emb_stage_model = None
#         else:
#             model = instantiate_from_config(config)
#             self.emb_stage_model = model
#             if not self.emb_stage_trainable:
#                 self.emb_stage_model.eval()
#                 # self.emb_stage_model.train()
#                 self.emb_stage_model.train = disabled_train

#     def get_xce(self, batch, N=None):  # dst_img,src_img(2,3,208,368);R(2,3,3);t(2,3))
#         xdict = dict()
#         for k, v in self.first_stage_key.items():
#             xdict[k] = self.get_input(v, batch, heuristics=k == "x")[
#                 :N
#             ]  # (2,3,208,368)

#         cdict = dict()
#         for k, v in self.cond_stage_key.items():
#             cdict[k] = self.get_input(v, batch, heuristics=k == "c")[
#                 :N
#             ]  # (2,3,208,368)

#         edict = dict()
#         for k, v in self.emb_stage_key.items():
#             edict[k] = self.get_input(v, batch, heuristics=False)[:N]
#             if k == "t":
#                 print(f"getxve value: {edict['t']}")


#         return xdict, cdict, edict  # (4,3,208,368)
class CodeBookEmbedder(nn.Module):
    def __init__(
        self,
        new_stage_config,
        emb_stage_config,
        emb_stage_key="camera",
        emb_stage_trainable=False,
        first_stage_key="dst_img",
        cond_stage_key="src_img",
        double_condition=False,
        no_embedding=True,  # 임베딩없이 원본 만드는놈
    ):
        super().__init__()
        # new_stage_config (첫 번째 모델) 초기화
        self.init_first_stage_from_ckpt(new_stage_config)
        # emb_stage_config (두 번째 모델) 초기화
        # self.device = "cuda"
        self.embedding = nn.Embedding(16384, 598)
        self.emb_stage_key = emb_stage_key
        self.emb_stage_trainable = emb_stage_trainable
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.double_condition = double_condition
        self.no_embedding = no_embedding
        self._midas = Midas()
        self._midas.eval()
        self._midas.train = disabled_train
        self.init_emb_stage_from_ckpt(emb_stage_config)
        # self.condition = nn.Linear(598 * 2, 598)

    def quantile_transform_per_set(self, tensor, n_quantiles=100):
        # 입력 텐서의 shape: (64, 1, 598)
        batch_size, _, seq_length = tensor.shape

        # 결과를 저장할 텐서 초기화
        transformed = torch.zeros_like(tensor)

        for i in range(batch_size):
            # 각 세트(1, 1, 598)에 대해 처리
            set_data = tensor[i, 0, :]  # shape: (598,)
            set_data = set_data.float()
            # 정렬된 데이터와 원본 인덱스 얻기
            sorted_data, sort_indices = torch.sort(set_data)

            # 균일 분포에서 n_quantiles+1개의 포인트 생성
            uniform_samples = Uniform(0, 1).sample((n_quantiles + 1,))
            uniform_samples = uniform_samples.sort().values

            # 정렬된 데이터를 균일하게 분포된 포인트에 매핑
            quantiles = torch.quantile(sorted_data, uniform_samples)

            # 원본 데이터의 각 값에 대해 가장 가까운 quantile 찾기
            transformed_values = (
                torch.searchsorted(quantiles, set_data).float() / n_quantiles
            )

            # 변환된 값을 결과 텐서에 저장
            transformed[i, 0, :] = transformed_values

        return transformed

    def init_first_stage_from_ckpt(self, config):
        """첫 번째 stage 모델 초기화"""
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

    def init_emb_stage_from_ckpt(self, config):
        """두 번째 stage 모델 초기화"""
        if config is None:
            self.emb_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.emb_stage_model = model
            if not self.emb_stage_trainable:
                self.emb_stage_model.eval()
                self.emb_stage_model.train = disabled_train

    def get_input(self, key, batch, heuristics=True):
        x = batch[key]
        if heuristics:
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if x.dtype == torch.double:
                x = x.float()
        return x

    @torch.no_grad()
    def get_normalized_c(
        self, cdict, edict, return_depth_only=False, fixed_scale=False, scale=None
    ):
        with torch.no_grad():
            quant_c, c_indices = self.encode_to_c(
                **cdict
            )  # (B,256,13,23),(B,299)   maxim(2,598)

            if not fixed_scale:
                scaled_idepth = self._midas.scaled_depth(
                    cdict["c"], edict.pop("points"), return_inverse_depth=True
                )
            else:
                scale = [0.18577382, 0.93059154] if scale is None else scale
                scaled_idepth = self._midas.fixed_scale_depth(
                    cdict["c"], return_inverse_depth=True
                )
            alpha = scaled_idepth.amax(dim=(1, 2))
            if 0.0 in alpha:
                alpha[alpha == 0.0] = 0.0001

            assert not torch.isnan(alpha).any()

            scaled_idepth = scaled_idepth / alpha[:, None, None]
            edict["t"] = edict["t"] * alpha[:, None]
            # quant_d, d_indices = self.encode_to_d(scaled_idepth[:,None,:,:]*2.0-1.0)

        # if return_depth_only:
        #     return d_indices, quant_d, scaled_idepth[:,None,:,:]*2.0-1.0
        embeddings = self.encode_to_e(**edict)  # (4,12,1024)

        return c_indices, embeddings

    @torch.no_grad()
    def encode_to_c(self, c):  # (B,3,208,368)
        quant_c, _, info = self.first_stage_model.encode(c)  # (4,256,13,23)
        indices = info[2].view(quant_c.shape[0], -1)  # (4,299)  maxim(4,598)
        return quant_c, indices

    # @torch.no_grad()
    # def encode_to_d(self, x):
    #     quant_z, _, info = self.depth_stage_model.encode(x)
    #     indices = info[2].view(quant_z.shape[0], -1)
    #     return quant_z, indices

    @torch.no_grad()
    def encode_to_e(self, **kwargs):
        return self.emb_stage_model(**kwargs)

    def normal(self, x, new_min=0.0, new_max=1.0):
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min

    def normalize(self, x):
        return x

    def normalize_per_set(self, input_tensor):
        # 입력 텐서의 shape: (64, 1, 598)
        batch_size, set_size, feature_dim = input_tensor.shape
        input_tensor = input_tensor.float()
        # 각 세트별로 평균과 표준편차 계산
        mean = input_tensor.mean(dim=2, keepdim=True)
        std = input_tensor.std(dim=2, keepdim=True)

        # 표준편차가 0인 경우 (상수 데이터) 처리
        std = torch.clamp(std, min=1e-8)

        # z-score 정규화 적용
        normalized = (input_tensor - mean) / std
        condition = torch.tensor(normalized)
        return condition

    def forward(self, c, e, key="camera"):
        # 여기서는 다시 짜야돔
        # batch안에는 이미지 전체가 들어있는거라서...
        # 딕셔너리의 모든 텐서 값을 self.device로 옮김
        # c = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in c.items()}
        # e = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in e.items()}

        # _, z_indices = self.encode_to_z(x)  # (B,299)
        c_indices, embeddings = self.get_normalized_c(
            c, e
        )  # 64, 598    , 64, 30, 102        c_indices = torch.unsqueeze(c_indices, 1)  # .to(self.device)
        c_indices = c_indices.unsqueeze(1)
        if self.no_embedding:  # 임베딩 없이가는거
            # condition = self.normalize_per_set(c_indices)
            # condition = self.embedding(c_indices)
            c_indices = c_indices.float()
            return c_indices
        else:
            if self.double_condition == True:
                condition = self.normal(c_indices)
                embeddings = self.normal(embeddings)
                return [condition, embeddings]
            else:
                condition = torch.cat((c_indices, embeddings), dim=2)
                return condition
        # 그냥 codebook만 주고 원본 복원
        # condition = c_indices + embeddings  # .to(self.device)
        # condition = self.condition(condition)
        # 정규화 코드
        # c_indices = self.normal(c_indices)
        # condition = c_indices + embeddings  # 6000 , 0.7
        # batch안에는 모든 데이터셋이 딕셔너리 형태로 들어있음
        # 잘 빼서 연산 수행 후 condition값인 64, 1, 598만 알아서 리턴해주면댐
        # c랑 e 합친거를 condition으로 줌


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device="cuda",
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(
        self,
        version="ViT-L/14",
        device="cuda",
        max_length=77,
        n_repeat=1,
        normalize=True,
    ):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, "b 1 d -> b k d", k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
        self,
        model,
        jit=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))
