import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Read the CSV file
res_dir = '../results/'
filename = 'runtime_8iVFBv2_redandblack.csv'
filepath = os.path.join(res_dir, filename)
df = pd.read_csv(filepath)

match = re.match(r'runtime_(.*?)_(.*?)\.csv', filename)
if match:
    dataset, sequence = match.groups()
else:
    print("Filename format not matched")

# Runtime columns to plot
runtime_cols = [
    'RAHT_prelude_time',
    'RAHT_transform_time',
    'order_RAGFT_time',
    # 'Quant_time',
    'Entropy_enc_time',
    'Entropy_dec_time',
    # 'Dequant_time',
    'iRAHT_time'
]

legend_names = {
    'RAHT_prelude_time': 'RAHT Prelude',
    'RAHT_transform_time': 'RAHT Transform',
    'order_RAGFT_time': 'Order_RAGFT',
    'Quant_time': 'Quantization',
    'Entropy_enc_time': 'Entropy Encoding',
    'Entropy_dec_time': 'Entropy Decoding',
    'Dequant_time': 'Dequantization',
    'iRAHT_time': 'Inverse RAHT'
}

# Group by Quantization_Step and calculate mean for each runtime column
grouped = df.groupby('Quantization_Step')[runtime_cols].mean()

total_times = grouped[runtime_cols].sum(axis=1)

# Convert to milliseconds
grouped = grouped * 1000
total_times = total_times * 1000

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Number of quantization steps and runtime types
quant_steps = grouped.index.values
n_steps = len(quant_steps)
n_runtimes = len(runtime_cols)

# Set bar width and positions
bar_width = 0.8 / n_runtimes
x = np.arange(n_steps)

# Color palette
colors = plt.cm.Set3(np.linspace(0, 1, n_runtimes))

# Plot bars for each runtime type
for i, col in enumerate(runtime_cols):
    offset = (i - n_runtimes/2) * bar_width + bar_width/2
    label = legend_names[col]
    ax.bar(x + offset, grouped[col], bar_width, label=label, color=colors[i])

# Add total time as text above each group of bars
ymax = max([max(grouped[col]) for col in runtime_cols])
for i, (step, total) in enumerate(zip(quant_steps, total_times)):
    ax.text(i, ymax * 0.85, f'{total:.2f}ms', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the plot
ax.set_xlabel('Quantization Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Runtime (milliseconds)', fontsize=12, fontweight='bold')
ax.set_title(f'{dataset}/{sequence} Runtime', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quant_steps)
ax.legend(loc='upper right', fontsize=12, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the plot
output_file = os.path.join(res_dir, f'runtime_{dataset}_{sequence}.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_file}")
plt.close()



# -------- Per-frame plots  --------
if 'Frame' in df.columns:
    for frame, df_f in df.groupby('Frame'):
        if frame != 1:
            continue
        grouped_f = df_f.groupby('Quantization_Step')[runtime_cols].mean()
        total_times_f = grouped_f[runtime_cols].sum(axis=1)

        # Convert to milliseconds
        grouped_f = grouped_f * 1000
        total_times_f = total_times_f * 1000

        fig, ax = plt.subplots(figsize=(14, 8))

        quant_steps_f = grouped_f.index.values
        n_steps_f = len(quant_steps_f)
        n_runtimes = len(runtime_cols)

        bar_width = 0.8 / n_runtimes
        x = np.arange(n_steps_f)

        # Reuse same colors
        for i, col in enumerate(runtime_cols):
            offset = (i - n_runtimes/2) * bar_width + bar_width/2
            label = legend_names[col]
            ax.bar(x + offset, grouped_f[col], bar_width, label=label, color=colors[i])

        ymax_f = max([grouped_f[col].max() for col in runtime_cols]) if n_steps_f else 0.0
        for i, (step, total) in enumerate(zip(quant_steps_f, total_times_f)):
            ax.text(i, ymax_f * 0.85, f'{total:.2f}ms',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('Quantization Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (milliseconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset}/{sequence} Runtime — Frame {frame}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(quant_steps_f)
        ax.legend(loc='upper right', fontsize=12, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        output_file = os.path.join(res_dir, f'runtime_{dataset}_{sequence}_{frame}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_file}")
        plt.close()