import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# ==================== First Chart: Cause Analysis ====================
# Data setup
categories1 = ['MLX', 'MindSpore', 'OneFlow', 'Total']
ec_data = [4, 0, 0, 4]
ni_data = [17, 25, 53, 95]
ld_data = [4, 0, 3, 7]
dbi_data = [8, 1, 1, 10]
mpc_data = [7, 4, 28, 39]

# Morandi color scheme - Cause chart (unified color system - slightly cool tones)
colors1 = ['#A8B5C0', '#B5A8C0', '#D0C0B0', '#A8C0B5', '#C0A8A8', '#C0A8B5']

# Set bar chart width
bar_width = 0.5
x1 = np.arange(len(categories1))

# Create stacked bar chart
bars11 = ax1.bar(x1, ec_data, bar_width, label='EC', color=colors1[0])
bars12 = ax1.bar(x1, ni_data, bar_width, bottom=ec_data, label='NI', color=colors1[1])
bars13 = ax1.bar(x1, ld_data, bar_width, bottom=np.array(ec_data) + np.array(ni_data), label='LD', color=colors1[3])
bars14 = ax1.bar(x1, dbi_data, bar_width, bottom=np.array(ec_data) + np.array(ni_data) + np.array(ld_data), label='DBI', color=colors1[2])
bars15 = ax1.bar(x1, mpc_data, bar_width, bottom=np.array(ec_data) + np.array(ni_data) + np.array(ld_data) + np.array(dbi_data), label='MPC', color=colors1[4])

# Set axes and styles for the first chart
ax1.set_title('(a) Cause Analysis', fontsize=28, fontweight='bold', pad=20)
ax1.set_ylabel('Counts', fontsize=24, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(categories1, fontsize=24, fontweight='bold')
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# Set legend for the first chart
ax1.legend(loc='upper left', fontsize=20, frameon=True, fancybox=True, shadow=True)

# ==================== Second Chart: Symptom Analysis ====================
# Data setup
categories2 = ['MLX', 'Mindspore', 'Oneflow', 'Total']
crash_data = [10, 0, 34, 44]
cpu_gpu_data = [8, 20, 33, 61]
src_tar_data = [22, 10, 18, 50]

# Morandi color scheme - Symptom chart (unified color system - slightly warm tones)
colors2 = ['#B0A8B0', '#B5A8A8', '#A8B5B5']

# Set bar chart width
x2 = np.arange(len(categories2))

# Create stacked bar chart
bars21 = ax2.bar(x2, crash_data, bar_width, label='Crash', color=colors2[0])
bars22 = ax2.bar(x2, cpu_gpu_data, bar_width, bottom=crash_data, label='CPU/GPU', color=colors2[1])
bars23 = ax2.bar(x2, src_tar_data, bar_width, bottom=np.array(crash_data) + np.array(cpu_gpu_data), label='Src libs/Tar libs', color=colors2[2])

# Set axes and styles for the second chart
ax2.set_title('(b) Symptom Analysis', fontsize=28, fontweight='bold', pad=20)
ax2.set_ylabel('Counts', fontsize=24, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(categories2, fontsize=24, fontweight='bold')
ax2.tick_params(axis='y', labelsize=18)
ax2.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# Set legend for the second chart
ax2.legend(loc='upper left', fontsize=20, frameon=True, fancybox=True, shadow=True)

# Adjust layout
plt.tight_layout()

# Display chart
# plt.show()

# Save image
plt.savefig('combined_charts.png', dpi=300, bbox_inches='tight') 