import os.path as osp
import os
import datetime
import time
from collections import OrderedDict
from einops import rearrange
from collections import defaultdict
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision import transforms

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.distributed as dist

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))


class domain_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768, 256) for _ in range (12))
        self.linear2 = nn.ModuleList(nn.Linear(256, 512) for _ in range (12))
        self.adain=AdaIN()
        self.gap=nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_cat = self.gap(x_cat).squeeze(1)
            x_out = self.linear1[i](x_cat)
            x_final = self.linear2[i](x_out)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)

        return output


class image_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(768, 512) for _ in range (12))
        self.adain=AdaIN()
        self.lin = nn.Linear(12,1)
        self.gap=nn.AdaptiveAvgPool2d((1,768))

    def forward(self, data, n_imgctx):
        data_prompt=[]
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(1)
            x_lin = self.linear[i](x_gap)
            data_prompt.append(x_lin)
        feat = torch.stack(data_prompt, dim=1)
        output = []
        for i in range(n_imgctx):
            x = self.lin(feat.permute(0,2,1))
            x = x.permute(0,2,1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1).squeeze(2)
        return feat_tokens


class style_mapping_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768, 384) for _ in range(12))
        self.linear2 = nn.ModuleList(nn.Linear(384, 512) for _ in range(12))
        self.adain = AdaIN()
        self.relu = nn.ReLU()
        self.gap=nn.AdaptiveAvgPool1d((768))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_gap = self.gap(x_cat)
            x_out = self.linear1[i](x_gap)
            x_relu = self.relu(x_out)
            x_final = self.linear2[i](x_relu)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_ctx = 24 + n_imgctx

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.domain_tokens = domain_projector()
        self.image_tokens = image_projector()
        self.style_mapping_tokens = style_mapping_projector()

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        print(tokenized_prompts.shape)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]
        self.n_imgctx = n_imgctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts
    @autocast()

    def forward(self, data, scr_feat=None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_imgctx = self.n_imgctx

        domaintokens = self.domain_tokens(data)
        imagetokens = self.image_tokens(data, n_imgctx)
        if scr_feat is not None:
            scr_feat = scr_feat.unsqueeze(1).repeat(domaintokens.shape[0], 12, 1).to(domaintokens.device)
            tokens = torch.cat((scr_feat, domaintokens, imagetokens), dim=1)
        else:
            tokens = torch.cat((domaintokens, domaintokens, imagetokens), dim=1)

        prompts = []
        for tokens_i in tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        return prompts, domaintokens


class ScrectEncoder(nn.Module):
    """ IFT """

    def __init__(self, clip_model):
        super().__init__()

        input_dim = clip_model.text_projection.shape[1]
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8

        self.pre_project = nn.Sequential(
            nn.Linear(input_dim, pre_dim1),
            nn.LayerNorm(pre_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim1, pre_dim2),
            nn.LayerNorm(pre_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim2, input_dim)
        ).half()

        self.post_project = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        ).half()

        self.logit_scale = clip_model.logit_scale

    def forward(self, feat):
        feat = self.pre_project(feat)
        feat = self.post_project(feat)
        return feat


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.screct_encoder = ScrectEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = self.prompt_learner.n_cls
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]
        self.feat_bank = {}
        self.bank_file = "bank.json"
        self.cfg = cfg
    def _load_bank_dict(self):
        try:
            if os.path.isfile(self.bank_file):
                with open(self.bank_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[warn] load {self.bank_file} failed: {e}")
        return {}

    def _save_bank_dict(self, d):
        try:
            with open(self.bank_file, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[warn] save {self.bank_file} failed: {e}")


    def init_domain_bank(self, domain):
        size = (self.n_cls - 1) * self.K
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        cached = self._load_bank_dict()
        if domain in cached:
            vec = torch.tensor(cached[domain], device=device, dtype=dtype)
            if vec.dim() > 1:
                vec = vec.view(-1)
            self.feat_bank[domain] = vec
            print(f"[bank] loaded '{domain}' from {self.bank_file} (shape={tuple(vec.shape)})")
            return
        self._cur_domain = domain
        self._domain_key_dict = {i: i for i in range(size)}
        self._domain_max_probs_list = [0.0 for _ in range(size)]
        self._domain_feat_bank = torch.zeros((size, self.dim), device=device, dtype=dtype)

    def update_domain_bank(self, logits, features, labels):
        logits = logits.detach()[:, :-1]
        features = features.detach()
        pseudo_label = torch.softmax(logits, dim=-1)
        max_probs, label_p = torch.max(pseudo_label, dim=-1)
        for i, l in enumerate(labels):
            if l == label_p[i]:
                index = int(l.item()) * self.K
                sub = self._domain_max_probs_list[index: index + self.K]
                if float(max_probs[i].item()) > min(sub):
                    min_idx_local = sub.index(min(sub))
                    global_idx = index + min_idx_local
                    self._domain_max_probs_list[global_idx] = float(max_probs[i].item())
                    self._domain_feat_bank[global_idx] = features[i].to(self._domain_feat_bank.device)
                    self._domain_key_dict[global_idx] = int(label_p[i].item())

    def finalize_domain_bank(self):
        valid_indices = [i for i, p in enumerate(self._domain_max_probs_list) if p != 0]
        if len(valid_indices) > 0:
            local_feats = self._domain_feat_bank[valid_indices]
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            print(self._cur_domain)
            if world_size > 1:
                gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
                dist.all_gather(gathered, local_feats)
                all_feats = torch.cat(gathered, dim=0)
            else:
                all_feats = local_feats
            self.feat_bank[self._cur_domain] = all_feats.mean(dim=0)
        else:
            self.feat_bank[self._cur_domain] = torch.zeros(self.dim,
                                                           device=self._domain_feat_bank.device,
                                                           dtype=self._domain_feat_bank.dtype)
        print("len of valid", len(valid_indices))

        vec_cpu = self.feat_bank[self._cur_domain].detach().float().cpu().tolist()
        cached = self._load_bank_dict()
        cached[self._cur_domain] = vec_cpu
        self._save_bank_dict(cached)
        print(f"[bank] saved '{self._cur_domain}' to {self.bank_file} (len={len(vec_cpu)})")

        del self._domain_key_dict
        del self._domain_max_probs_list
        del self._domain_feat_bank
        del self._cur_domain

    @autocast()
    def forward(self, s_image, t_image=None, e_image=None, domain=None, mode=None):
        if mode == "testing":
            image = s_image
            image_features, data = self.image_encoder(image.type(self.dtype))
            screct_domain = self.screct_encoder(self.feat_bank[domain].unsqueeze(0).to(data.device))
            prompts, domaintokens = self.prompt_learner(data, screct_domain)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.encode_text_features(prompts)
            logits = self.compute_logits(text_features, image_features)
            return logits

        elif mode == "constructing":
            image = s_image
            image_features, data = self.image_encoder(image.type(self.dtype))
            prompts, domaintokens = self.prompt_learner(data)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.encode_text_features(prompts)
            logits = self.compute_logits(text_features, image_features)
            return logits, image_features, domaintokens, text_features

        elif mode == "training":
            source_image_features, source_data = self.image_encoder(s_image.type(self.dtype))
            target_image_features, target_data = self.image_encoder(t_image.type(self.dtype))
            expend_image_features, expend_data = self.image_encoder(e_image.type(self.dtype))

            screct_author = self.screct_encoder(self.feat_bank[self.cfg.DATASET.SOURCE_DOMAINS[0]].unsqueeze(0).to(source_data.device))

            source_prompts_with_author_scr, source_domaintokens = self.prompt_learner(source_data, screct_author)
            target_prompts_with_author_scr, target_domaintokens = self.prompt_learner(target_data, screct_author)
            expend_prompts_with_author_scr, expend_domaintokens = self.prompt_learner(expend_data, screct_author)

            source_image_features = source_image_features / source_image_features.norm(dim=-1, keepdim=True)
            target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)
            expend_image_features = expend_image_features / expend_image_features.norm(dim=-1, keepdim=True)

            source_text_features_with_author_scr = self.encode_text_features(source_prompts_with_author_scr)
            target_text_features_with_author_scr = self.encode_text_features(target_prompts_with_author_scr)
            expend_text_features_with_author_scr = self.encode_text_features(expend_prompts_with_author_scr)

            logits_source_with_author_scr = self.compute_logits(source_text_features_with_author_scr, source_image_features)
            logits_target_with_author_scr = self.compute_logits(target_text_features_with_author_scr, target_image_features)
            logits_expend_with_author_scr = self.compute_logits(expend_text_features_with_author_scr, expend_image_features)

            return logits_source_with_author_scr, logits_target_with_author_scr, logits_expend_with_author_scr, source_text_features_with_author_scr, expend_text_features_with_author_scr



        else:
            print("Error mode")

    def encode_text_features(self, prompts):
        features = [self.text_encoder(p, self.tokenized_prompts) for p in prompts]
        features = torch.stack(features)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return features

    def check(self, *pairs):
        for name, t in pairs:
            if t is None:
                print(f"{name}: None")
                continue
            t_flat = t.float().view(-1)
            t_nonan = t_flat[~torch.isnan(t_flat)]
            min_val = float(t_nonan.min()) if t_nonan.numel() > 0 else None
            max_val = float(t_nonan.max()) if t_nonan.numel() > 0 else None
            print(f"{name}: shape={getattr(t, 'shape', None)} device={t.device} dtype={t.dtype} "
                  f"has_nan={torch.isnan(t).any().item()} has_inf={torch.isinf(t).any().item()} "
                  f"min={min_val} max={max_val}")

    def compute_logits(self, text_features, image_features):
        out = []
        scale = self.logit_scale.exp()
        for txt, im in zip(text_features, image_features):
            im_f = im.float()
            txt_f = txt.float()
            l_i = (scale.float() * (im_f @ txt_f.t()))
            out.append(l_i.to(image_features.dtype))
        return torch.stack(out)


class entropy_loss(nn.Module):
	def __init__(self):
		super(entropy_loss, self).__init__()

	def forward(self, target_prob):
		full_enp = torch.zeros(target_prob.shape[0])
		target_prob = nn.functional.normalize(target_prob, dim=0)

		for i in range(len(target_prob)):
			total_en = 0
			for j in range(target_prob.shape[1]):
				total_en = total_en - target_prob[i][j] * torch.log(target_prob[i][j] + 1e-8)
			full_enp[i] = total_en
		avg_full_enp = torch.mean(full_enp)
		return avg_full_enp


class AdaClipModel(torch.nn.Module):
    def __init__(self, model, num_classes=1000):
        super(AdaClipModel, self).__init__()
        self.model = model
        self.visual_encoder = self.model

        output_dim = self.visual_encoder.output_dim
        self.fc = torch.nn.Linear(output_dim, 2)

    def forward(self, image):
        x, data = self.visual_encoder(image)
        x = self.fc(x)
        return x

def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

@TRAINER_REGISTRY.register()
class IPCLIPB16(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IPCLIPB16.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print('******************************************')
        classnames.append('unauthorized')
        print('classnames', classnames, len(classnames))
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.dim = clip_model.text_projection.shape[1]

        if cfg.TRAINER.IPCLIPB16.PREC == "fp32" or cfg.TRAINER.IPCLIPB16.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls

        name_to_update = ["prompt_learner", "screct_encoder"]

        for name, param in self.model.named_parameters():
            if not any(kw in name for kw in name_to_update):
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        len_train_loader_e = len(self.train_loader_e)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u, len_train_loader_e)
        else:
            raise ValueError

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IPCLIPB16.PREC == "amp" else None
        self.construct_bank_before_training()

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def construct_bank_before_training(self):
        print("1.construct_bank before training")
        self.set_model_mode("eval")

        domain_loaders = {
            self.cfg.DATASET.SOURCE_DOMAINS[0]: self.train_loader_x,
            self.cfg.DATASET.SOURCE_DOMAINS[0] + "_expend": self.train_loader_e,
            self.cfg.DATASET.TARGET_DOMAINS[0]: self.train_loader_u,
        }

        for domain, loader in domain_loaders.items():
            self.model.init_domain_bank(domain)
            cache_hit = (domain in self.model.feat_bank) and (not hasattr(self.model, "_domain_feat_bank"))
            if cache_hit:
                print(f"[bank] hit cache for '{domain}', skip constructing loop.")
                continue
            for batch_idx, batch in enumerate(loader):
                inputs, labels = self.parse_batch_test(batch)
                logits, features, _, _ = self.model(inputs, domain=domain, mode="constructing")
                self.model.update_domain_bank(logits, features, labels)
            self.model.finalize_domain_bank()
            print(f"{domain.capitalize()} feature banks are completed!")


    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()
        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )


    def train(self):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        len_train_loader_e = len(self.train_loader_e)

        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u, len_train_loader_e)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)
        train_loader_e_iter = iter(self.train_loader_e)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            try:
                batch_e = next(train_loader_e_iter)
            except StopIteration:
                train_loader_e_iter = iter(self.train_loader_e)
                batch_e = next(train_loader_e_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u, batch_e)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def forward_backward(self, batch_s, batch_t, batch_e):
        self.entropy = entropy_loss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        image_s, label_s, image_t, label_t, image_e, label_e = self.parse_batch_train(batch_s, batch_t, batch_e)

        prec = self.cfg.TRAINER.IPCLIPB16.PREC
        if prec == "amp":
            with autocast():
                logits_source_with_author_scr, logits_target_with_author_scr, logits_expend_with_author_scr, source_text_features_with_author_scr, expend_text_features_with_author_scr = self.model(image_s, image_t, image_e, mode='training')
                loss_ce_s_with_a_scr = F.cross_entropy(logits_source_with_author_scr, label_s)
                label_u = torch.full_like(label_s, self.n_cls-1)
                loss_ce_s_with_a_scr_u = F.cross_entropy(logits_source_with_author_scr, label_u)
                loss_ce_t_with_a_scr = F.cross_entropy(logits_target_with_author_scr, label_u)
                loss_ce_e_with_a_scr = F.cross_entropy(logits_expend_with_author_scr, label_u)

                source_textfeat_a = F.log_softmax(source_text_features_with_author_scr, dim=1)
                expend_textfeat_a = F.softmax(expend_text_features_with_author_scr, dim=1)
                loss_kl_ae = kl_loss(source_textfeat_a, expend_textfeat_a)


                up_bounder = 5
                if loss_kl_ae > up_bounder:
                    loss_kl_ae = torch.clamp(loss_kl_ae, 0, up_bounder)
                loss = loss_ce_s_with_a_scr - 0.1 * loss_ce_s_with_a_scr_u + loss_ce_t_with_a_scr + 0.1 * loss_ce_e_with_a_scr - loss_kl_ae

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()


        loss_summary = {
            "acc_x":
            compute_accuracy(logits_source_with_author_scr[:, :self.n_cls], label_s)[0].item(),
            "loss": loss.item(),
            "loss_ce_s_with_a_scr": loss_ce_s_with_a_scr.item(),
            "loss_ce_t_with_a_scr": loss_ce_t_with_a_scr.item(),
            "loss_ce_e_with_a_scr": loss_ce_e_with_a_scr.item(),
            "loss_ce_s_with_a_scr_u": loss_ce_s_with_a_scr_u.item(),
            "loss_kl_ae": loss_kl_ae.item(),
        }

        self.update_lr()

        return loss_summary


    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            metric_file_path = os.path.join(self.output_dir, "metric.txt")
            domain_scr = self.cfg.DATASET.SOURCE_DOMAINS[0]
            with open(metric_file_path, "a") as f:
                results = self.test(domain_scr)
                f.write(f"epoch: {self.epoch}: {domain_scr}\n")
                f.write(domain_scr + " | " + " | ".join([f"{domain}: {metrics}" for domain, metrics in results.items()]) + "\n")

            self.save_model(self.epoch, self.output_dir, model_name="model--{}".format(self.epoch))

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch_s, batch_t, batch_e):
        input_s = batch_s["img"]
        label_s = batch_s["label"]
        input_t = batch_t["img"]
        label_t = batch_t["label"]
        input_e = batch_e["img"]
        label_e = batch_e["label"]

        input_s = input_s.to(self.device)
        label_s = label_s.to(self.device)
        input_t = input_t.to(self.device)
        label_t = label_t.to(self.device)
        input_e = input_e.to(self.device)
        label_e = label_e.to(self.device)
        return input_s, label_s, input_t, label_t, input_e, label_e


    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)


    @torch.no_grad()
    def test(self, domain_scr):

        self.set_model_mode("eval")
        test_loaders = {
            self.cfg.DATASET.SOURCE_DOMAINS[0]: self.test_loader_x,
            self.cfg.DATASET.TARGET_DOMAINS[0]: self.test_loader_u,
        }
        results_dict = defaultdict(dict)
        for domain, loader in test_loaders.items():
            self.evaluator.reset()
            print(f"Evaluate on the *{domain}* set for all class")
            for batch_idx, batch in enumerate(tqdm(loader)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input=input, domain=domain_scr, mode='testing')
                self.evaluator.process(output, label)
            results = self.evaluator.evaluate()
            for k, v in results.items():
                tag = f"{domain}/{k}"
                self.write_scalar(tag, v, self.epoch)
            results_dict[domain]["acc_all"] = list(results.values())[0]


            self.evaluator.reset()
            print(f"Evaluate on the *{domain}* set for unauthorized class")
            for batch_idx, batch in enumerate(tqdm(loader)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input=input, domain=domain_scr, mode='testing')
                label_au = torch.full_like(label, self.n_cls-1)
                self.evaluator.process(output, label_au)
            results = self.evaluator.evaluate()
            for k, v in results.items():
                tag = f"{domain}/{k}"
                self.write_scalar(tag, v, self.epoch)
            results_dict[domain]["acc_auth"] = list(results.values())[0]
        return results_dict