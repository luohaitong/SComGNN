import math

import torch


def metrics_at_k_2(pos_scores, neg_scores, k):
    # 合并正样本和负样本的得分
    scores = torch.cat([pos_scores, neg_scores], dim=1)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)

    ranking = torch.argsort(scores, dim=1, descending=True)
    ideal_ranking = torch.argsort(labels, dim=1, descending=True)
    ranking_k = ranking[:, :k]
    ideal_ranking_k = ideal_ranking[:, :k]

    match_score = torch.gather(labels, 1, ranking)
    index_matrix = torch.arange(1, match_score.shape[1] + 1, device=scores.device).unsqueeze(0).repeat(
        match_score.shape[0], 1)

    match_score = torch.mul(match_score, index_matrix)
    match_score[match_score == 0] = 1e9
    mrr = torch.mean(1.0 / match_score.float())
    hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))

    # 计算 NDCG@K

    # Compute the discounted cumulative gain (DCG)
    ranked_scores = torch.gather(labels, 1, ranking_k)
    discounts = torch.log2(torch.arange(2, ranked_scores.shape[1] + 2, device=scores.device))
    dcg = torch.sum((2 ** ranked_scores - 1) / discounts, dim=1)
    # Compute the ideal DCG (IDCG)
    ideal_scores = torch.gather(labels, 1, ideal_ranking_k)
    ideal_dcg = torch.sum((2 ** ideal_scores - 1) / discounts, dim=1)

    # Compute the NDCG
    ndcg = dcg / ideal_dcg
    ndcg = torch.mean(ndcg)

    return mrr, hr, ndcg

lht = torch.Tensor([[0.5]])
lht2 = torch.Tensor([[0.2, 0.9]])

mrr, hr, ndcg = metrics_at_k_2(lht, lht2, 3)
