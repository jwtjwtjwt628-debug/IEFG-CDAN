from torch import nn
import numpy as np
import torch

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class AdversarialNet(nn.Module):
    def __init__(self, in_feature, hidden_size,max_iter=10000.0, trade_off_adversarial='Step', lam_adversarial=1.0):
        super(AdversarialNet, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.__in_features = 1

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        elif self.trade_off_adversarial == 'Step':
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return self.__in_features

class AdversarialNet_multi(nn.Module):
    def __init__(self, in_feature, output_size,
                 hidden_size,max_iter=10000.0,
                 trade_off_adversarial='Step',
                 lam_adversarial=1.0
                 ):
        super(AdversarialNet_multi, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, output_size)
        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.__in_features = 1

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        elif self.trade_off_adversarial == 'Step':
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return self.__in_features
    
class FineGrainedAdversarialNet(nn.Module):
    """
    IEFG-CDAN: Information Entropy-aware Fine-Grained Conditional Domain Adversarial Network
    基于信息熵感知与细粒度对抗的跨工况故障诊断网络

    相比原版 CDANN-FG 的三项核心改进:
    (1) Boltzmann 熵权重: exp(-H/T) 替代 1+exp(-H)，拉大高/低置信样本的权重差距
    (2) 双阈值鲁棒过滤: 极低置信样本(max_prob < tau_low)彻底排除在细粒度对齐之外
    (3) 与训练循环中的渐进门控配合: 训练初期仅全局对齐，逐步过渡到细粒度对齐
    """
    def __init__(self, in_feature, output_size, hidden_size, num_classes,
                 max_iter=10000.0, trade_off_adversarial='Step', lam_adversarial=1.0,
                 entropy_temperature=1.0, tau_high=0.9, tau_low=0.5):
        super(FineGrainedAdversarialNet, self).__init__()
        self.num_classes = num_classes
        self.entropy_temperature = entropy_temperature
        self.tau_high = tau_high
        self.tau_low = tau_low

        self.domain_discriminators = nn.ModuleList([
            AdversarialNet_multi(in_feature, output_size, hidden_size,
                                max_iter, trade_off_adversarial, lam_adversarial)
            for _ in range(num_classes)
        ])

    def forward(self, features, softmax_preds, is_source_mask=None):
        """
        Args:
            features: (B, D) bottleneck特征
            softmax_preds: (B, num_fault_classes) 分类器softmax预测概率
            is_source_mask: (B,) bool, True=源域样本, None=旧版兼容(全部参与)
        Returns:
            final_ad_out: (B, output_size) 加权后的域判别logits
        """
        epsilon = 1e-5
        B = features.size(0)

        # --- (1) 信息熵计算 ---
        entropy = -torch.sum(softmax_preds * torch.log(softmax_preds + epsilon), dim=1)  # (B,)
        num_fault_classes = softmax_preds.size(1)
        H_normalized = entropy / np.log(num_fault_classes + epsilon)  # 归一化到 [0, 1]

        # --- (2) Boltzmann 熵权重: exp(-H/T), T越小区分度越大 ---
        entropy_weight = torch.exp(-H_normalized / self.entropy_temperature)  # (B,)

        # --- (3) 双阈值鲁棒过滤 ---
        max_prob, _ = softmax_preds.max(dim=1)  # (B,)
        threshold_weight = torch.ones(B, device=features.device)

        mask_high = max_prob >= self.tau_high
        mask_low = max_prob < self.tau_low
        mask_mid = (~mask_high) & (~mask_low)

        threshold_weight[mask_high] = 1.0
        threshold_weight[mask_mid] = (max_prob[mask_mid] - self.tau_low) / (self.tau_high - self.tau_low + 1e-8)
        threshold_weight[mask_low] = 0.0

        # 源域样本始终参与对齐（权重=1）
        if is_source_mask is not None:
            entropy_weight[is_source_mask] = 1.0
            threshold_weight[is_source_mask] = 1.0

        combined_weight = (entropy_weight * threshold_weight).detach().unsqueeze(1)  # (B, 1)

        # --- (4) 细粒度加权聚合 ---
        ad_out = []
        for k in range(self.num_classes):
            d_k_out = self.domain_discriminators[k](features)  # (B, output_size)
            class_weight = softmax_preds[:, k].detach().unsqueeze(1)  # (B, 1)
            weight = class_weight * combined_weight  # (B, 1)
            ad_out.append(d_k_out * weight)

        final_ad_out = torch.sum(torch.stack(ad_out, dim=-1), dim=-1)  # (B, output_size)

        return final_ad_out