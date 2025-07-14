import pandas as pd
import torch
from transformers import CLIPTokenizer, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# モデルとトークナイザーのロード
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# CSV読み込み
df = pd.read_csv("words.csv", header=None, names=["word1", "word2"])

# トークナイズ
all_texts = df["word1"].tolist() + df["word2"].tolist()
inputs = tokenizer(all_texts, padding=True, return_tensors="pt")

# 埋め込み取得
with torch.no_grad():
    outputs = model.get_text_features(**inputs)

# Δembedding 計算
n = len(df)
emb1 = outputs[:n, :]
emb2 = outputs[n:, :]
delta = emb1 - emb2

# コサイン類似度行列
sim_matrix = cosine_similarity(delta.cpu().numpy())

# ラベルリスト作成
labels = [f"{w1}–{w2}" for w1, w2 in zip(df["word1"], df["word2"])]

# プロット
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_matrix, interpolation='nearest', cmap='viridis')

# 軸目盛り＆ラベル設定
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("word1–word2 pairs")
ax.set_ylabel("word1–word2 pairs")
ax.set_title("ΔEmbedding Cosine Similarity Matrix")

# ───── セルへの数値注釈 ─────
# 値が濃い領域では文字色を白、それ以外は黒にする簡易的な例
threshold = sim_matrix.max() / 2.0
for i in range(n):
    for j in range(n):
        val = sim_matrix[i, j]
        color = "white" if val < threshold else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6)
# ───── 注釈ここまで ─────

# カラーバー
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
