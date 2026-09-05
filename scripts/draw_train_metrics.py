import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("excess/models/metrics_02.csv", comment='#')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('white')


axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=1.5, alpha=0.9)
axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=1.5, alpha=0.9)
axes[0].set_title('losses', fontsize=14, fontweight='bold')
axes[0].set_ylabel('loss')
axes[0].legend(loc='upper right')


axes[1].plot(df['epoch'], df['abs_rel'], color='#d62728', linewidth=1.5)
axes[1].set_title('absolute relative error', fontsize=14, fontweight='bold')
axes[1].set_ylabel('error')


axes[2].plot(df['epoch'], df['d1'], label='d1 (δ < 1.25)', color='#2ca02c', linewidth=1.5, alpha=0.9)
axes[2].plot(df['epoch'], df['d3'], label='d3 (δ < 1.25³)', color='#9467bd', linewidth=1.5, alpha=0.9)
axes[2].set_title('accuracy thresholds d1 & d3', fontsize=14, fontweight='bold')
axes[2].set_ylabel('accuracy')
axes[2].legend(loc='lower right')


for ax in axes:
    ax.set_xlabel('Epoch', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig('metrics_train_info.png', dpi=300, bbox_inches='tight')
plt.show()