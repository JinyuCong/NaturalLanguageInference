from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, \
    PreTrainedTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import NLIDataset
from models import ESIMModel
from datasets import load_dataset


class Critic(nn.Module):
    def __init__(self, base_model: ESIMModel, hidden_size):
        super(Critic, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, premise, hypothesis):
        a_emb = self.base_model.embedding(premise)
        b_emb = self.base_model.embedding(hypothesis)

        a_encoded, _ = self.base_model.encoder(a_emb)  # (batch_size, seq_len, 2*hidden_size)
        b_encoded, _ = self.base_model.encoder(b_emb)

        attn = torch.matmul(a_encoded, b_encoded.transpose(1, 2))

        a_tilde = torch.matmul(F.softmax(attn, dim=2), b_encoded)
        b_tilde = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2), a_encoded)

        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        v_a, _ = self.base_model.inference_encoder(m_a)
        v_b, _ = self.base_model.inference_encoder(m_b)

        v_a = torch.cat([v_a.max(dim=1)[0], v_a.mean(dim=1)], dim=1)
        v_b = torch.cat([v_b.max(dim=1)[0], v_b.mean(dim=1)], dim=1)
        v = torch.cat([v_a, v_b], dim=1)

        value = self.value_head(v).squeeze(-1)  # (batch_size,)
        return value


def reward_function(pred_logits, true_labels):
    probs = F.softmax(pred_logits, dim=-1)
    correct = (torch.argmax(probs, dim=-1) == true_labels).float()
    confidence = probs.gather(1, true_labels.unsqueeze(1)).squeeze()  # 正确类别的概率
    reward = correct * confidence  # 结合正确性和置信度
    return reward

'''
def reward_function(actions: torch.Tensor, true_labels: torch.Tensor):
    return (actions == true_labels).to(torch.float32)
'''


def compute_policy_loss(log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    return loss.mean(-1).mean()


def compute_value_loss(values, old_values, returns, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2

    return loss.mean(-1).mean()


@dataclass
class Samples:
    premises_ids: torch.Tensor
    hypotheses_ids: torch.Tensor
    predictions: torch.Tensor
    labels: torch.Tensor


@dataclass
class Experience:
    """
    prediciton是一个数，预测标签
    """
    premises_ids: torch.Tensor
    hypotheses_ids: torch.Tensor
    predictions: torch.Tensor
    actions: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    rewards: torch.Tensor
    kl: Optional[torch.Tensor] = None


def generate_samples(premises_ids,
                     hypotheses_ids,
                     gold_labels,
                     policy_model: ESIMModel):
    """
    Args:
        premises_ids: tensor格式的已经分好batch的多个premise
        hypotheses_ids: tensor格式的已经分好batch的多个hypothesis
        gold_labels: 这个batch的数据对应的labels
        policy_model: ESIM
    """
    policy_model.eval()
    premises_ids = premises_ids.to(device)
    hypotheses_ids = hypotheses_ids.to(device)
    gold_labels = gold_labels.to(device)

    logits = policy_model(premises_ids, hypotheses_ids)
    predicitons = logits.argmax(dim=-1)
    samples = Samples(
        premises_ids=premises_ids,
        hypotheses_ids=hypotheses_ids,
        predictions=predicitons,
        labels=gold_labels
    )

    return samples


def compute_rewards(actions,
                    logits,
                    true_labels,
                    action_log_probs,
                    ref_action_log_probs,
                    kl_ctl=0.01,
                    clip_reward_value=1.0):
    """
    Args:
        actions: int, 预测的label
        true_labels: int, 真实值label
        action_log_probs: 参数更新后的模型预测的label的概率
        ref_action_log_probs: 旧模型预测的label的概率
        kl_ctl: float, KL coefficient
    """
    r_ext = reward_function(logits, true_labels)

    kl = action_log_probs - ref_action_log_probs
    r_kl = -kl_ctl * kl

    rewards = r_ext + r_kl

    rewards = torch.clamp(rewards, min=-clip_reward_value, max=clip_reward_value)

    return rewards


def compute_advantages_and_returns(values: torch.Tensor, rewards: torch.Tensor):
    returns = rewards
    advantages = rewards - values.detach()

    return advantages, returns


def generate_experiences(samples):
    policy_model.eval()
    ref_model.eval()
    critic_model.eval()

    premises_ids = samples.premises_ids
    hypotheses_ids = samples.hypotheses_ids
    predictions = samples.predictions
    gold_labels = samples.labels
    with torch.no_grad():
        # 计算策略模型的最后选定的action的概率
        logits = policy_model(premises_ids, hypotheses_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        actions = torch.argmax(logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        # 计算ref模型的最后选定的action的概率
        ref_logits = ref_model(premises_ids, hypotheses_ids)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_actions = torch.argmax(ref_logits, dim=-1)
        ref_action_log_probs = ref_log_probs.gather(dim=-1, index=ref_actions.unsqueeze(-1)).squeeze(-1)
        # 计算价值
        values = critic_model(premises_ids, hypotheses_ids)
        # 计算奖励模型的奖励值
        rewards = compute_rewards(actions, logits, gold_labels, action_log_probs, ref_action_log_probs)
        # 计算优势和回报
        advantages, returns = compute_advantages_and_returns(values, rewards)

    experience = Experience(
        premises_ids=premises_ids,
        hypotheses_ids=hypotheses_ids,
        predictions=predictions,
        actions=actions,
        action_log_probs=action_log_probs,
        values=values,
        returns=returns,
        advantages=advantages,
        rewards=rewards
    )

    return experience


def train_step(experience: Experience, steps):
    policy_model.train()
    policy_optimizer.zero_grad()

    premises_ids = experience.premises_ids
    hypotheses_ids = experience.hypotheses_ids
    old_actions = experience.actions
    old_action_log_probs = experience.action_log_probs
    old_values = experience.values
    returns = experience.returns
    advantages = experience.advantages

    logits = policy_model(premises_ids, hypotheses_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    #actions = torch.argmax(logits, dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=old_actions.unsqueeze(-1)).squeeze(-1)

    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages)
    policy_loss.backward()
    policy_optimizer.step()

    critic_model.train()
    critic_optimizer.zero_grad()
    values = critic_model(premises_ids, hypotheses_ids)
    value_loss = compute_value_loss(values, old_values, returns)
    value_loss.backward()
    critic_optimizer.step()

    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")


def train():
    steps = 0

    for episode in range(episodes):
        for pre_ids, pre_mask, hypo_ids, hypo_mask, labels in train_loader:
            # 生成样本
            sample = generate_samples(pre_ids, hypo_ids, labels, policy_model)
            # 生成经验（获取优势，奖励，回报等）
            experience = generate_experiences(sample)

            for epoch in range(max_epochs):
                train_step(experience, steps)
                steps += 1

            torch.cuda.empty_cache()

        test_correct, test_total = 0, 0
        policy_model.eval()
        with torch.no_grad():
            for test_pre_ids, test_pre_mask, test_hypo_ids, test_hypo_mask, test_labels in test_loader:
                test_pre_ids, test_hypo_ids, test_labels = test_pre_ids.to(device), test_hypo_ids.to(device), test_labels.to(device)
                logits = policy_model(test_pre_ids, test_hypo_ids)
                prediciton = torch.argmax(logits, dim=-1)
                test_correct += (prediciton == test_labels).sum().item()
                test_total += test_labels.size(0)

        test_accuracy = test_correct / test_total
        print(f"episode [{episode+1}|{episodes}] test accuracy: {test_accuracy*100:4f}%")


if __name__ == "__main__":
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # premise和hypothesis的最长长度
    max_length = 64
    batch_size = 32
    embedding_dim = 128
    hidden_size = 128
    lr = 1e-5

    snli_dataset = load_dataset("snli")
    text_train_dataset = snli_dataset["train"]
    text_test_dataset = snli_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NLIDataset(text_train_dataset, tokenizer, max_length)
    test_dataset = NLIDataset(text_test_dataset, tokenizer, max_length)

    # 定义dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 策略模型
    policy_model = ESIMModel(vocab_size, embedding_dim, hidden_size).to(device)
    policy_model.load_state_dict(torch.load("./weights/esim_snli.pth"))
    # 参考模型
    ref_model = ESIMModel(vocab_size, embedding_dim, hidden_size).to(device)
    ref_model.load_state_dict(torch.load("./weights/esim_snli.pth"))
    # 价值模型
    critic_model = Critic(policy_model, hidden_size).to(device)

    # 初始化优化器
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=lr)

    train()
