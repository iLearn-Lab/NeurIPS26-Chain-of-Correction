import matplotlib.pyplot as plt
import numpy as np

# --- 1. Global Academic Formatting Settings --- 
plt.rcParams['pdf.fonttype'] = 42     # Force PDF to embed Type 42 (TrueType) fonts, avoiding Type 3
plt.rcParams['ps.fonttype'] = 42      # Apply the same for PostScript compatibility
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12.5

# --- 2. Core Data ---
categories = ['Setup Chess', 'Move Hanger', 'Money in Safe', 'Wine at Rack', 'Meat on Grill', 'Turn Oven On', 'Open Drawer']
N = len(categories)

base_mean = [6.8, 37.9, 72.8, 80.3, 83.9, 91.9, 89.9]
forewarn_mean = [6.9, 1.1, 69.4, 47.8, 67.5, 91.9, 42.1]
lpb_mean = [8.8, 38.8, 72.8, 78.1, 84.6, 92.1, 86.8]
gpc_mean = [8.7, 37.5, 72.2, 81.6, 83.8, 92.3, 90.8]
ours_mean = [9.0, 39.2, 82.0, 89.8, 91.3, 92.4, 92.7]

# Standard Deviation Data
base_std = [4.1, 18.9, 21.7, 21.5, 28.2, 4.1, 13.6]
forewarn_std = [3.3, 1.7, 20.7, 37.2, 22.6, 5.5, 30.5]
lpb_std = [5.5, 19.3, 23.6, 21.8, 28.4, 4.5, 11.9]
gpc_std = [7.1, 19.2, 24.2, 20.8, 27.5, 3.3, 8.5]
ours_std = [4.8, 20.2, 8.5, 8.4, 7.8, 6.3, 3.2]

# Professional Color Palette
colors = [
    '#A0A2A6',  # Base Policy (Neutral Gray)
    '#FCD5A1',  # FOREWARN (Light Peach)
    '#F6B26B',  # LPB (Soft Warm Orange)
    '#A8C3A6',  # GPC (Sage Green)
    '#95C6F7'   # Ours (Clear Sky Blue)
]

# Darker Error Bar Colors to anchor visual focus on light bars
err_colors = [
    '#6F7175',  # Base Dark Gray
    '#D1A169',  # FOREWARN Deep Peach
    '#C47A33',  # LPB Dark Warm Orange
    '#7D9C7B',  # GPC Deep Grass Green
    '#5A9FD4'   # Ours Dark Sky Blue
]

# --- 3. Canvas and Layout ---
fig, ax = plt.subplots(figsize=(15, 4.5))

x = np.arange(N)
width_step = 0.17
width = 0.14

error_kw = {
    'capsize': 3.5,           # Cap width
    'elinewidth': 1,          # Error line thickness
    'capthick': 1,            # Cap thickness
    'zorder': 4               # Ensure error bars are on top
}

# --- 4. Plotting Grouped Bar Chart ---
bar_kwargs = {'linewidth': 0, 'zorder': 3}  # Linewidth handled independently for clarity

ax.bar(x - 2*width_step, base_mean, width, label='Base Policy', \
       color=colors[0], yerr=base_std, error_kw={'ecolor': err_colors[0], **error_kw}, **bar_kwargs)
ax.bar(x - width_step, forewarn_mean, width, label='FOREWARN', \
       color=colors[1], yerr=forewarn_std, error_kw={'ecolor': err_colors[1], **error_kw}, **bar_kwargs)
ax.bar(x, lpb_mean, width, label='LPB', \
       color=colors[2], yerr=lpb_std, error_kw={'ecolor': err_colors[2], **error_kw}, **bar_kwargs)
ax.bar(x + width_step, gpc_mean, width, label='GPC', \
       color=colors[3], yerr=gpc_std, error_kw={'ecolor': err_colors[3], **error_kw}, **bar_kwargs)
ax.bar(x + 2*width_step, ours_mean, width, label='Ours', \
       color=colors[4], yerr=ours_std, error_kw={'ecolor': err_colors[4], **error_kw}, **bar_kwargs)

# Annotating values for 'Ours'
for i, (v, err) in enumerate(zip(ours_mean, ours_std)):
    # Dynamic offset: error value + 2, with a minimum offset of 5 for visibility
    offset = max(err + 2, 5)
    ax.text(i + 2*width_step, v + offset, f'{v:.1f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#2C3E50')

ax.set_xlim(-0.52, N - 0.5)

# --- 5. Decorations and Detail Optimization ---
# Enable horizontal grid lines with transparency
ax.yaxis.grid(True, linestyle='--', color='#E0E0E0', alpha=0.5, zorder=0)

# Formatting labels and ticks
ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', labelpad=0.2)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha='center', fontweight='bold', fontsize=16)

# Set axis limits and intervals
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

# Fine-tuning Spines: Removing top/right and setting thickness for bottom/left
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['left'].set_color('#4A4A4A')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['bottom'].set_color('#4A4A4A')
ax.spines['bottom'].set_zorder(10)

ax.tick_params(axis='both', colors='#4A4A4A', width=1.0)

legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=5,
                   frameon=True, fancybox=True, framealpha=0.95,
                   edgecolor='#B0B0B0', facecolor='#F8F9FA',
                   fontsize=16, prop={'weight': 'bold'})
legend.get_frame().set_linewidth(0.8)
 
plt.tight_layout()
plt.savefig('per_task_results.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
# plt.show()