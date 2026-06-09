import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

fig = plt.figure(figsize=(6.8, 2.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

headers = ['#', 'Model', 'Estimation']
data = [
    ['1', 'GLM-MRP',         'Frequentist MLE, fixed effects'],
    ['2', 'GLMER-MRP',       'Frequentist MLE Multilevel, random intercepts'],
    ['3', 'GLMER-Stan',      'Bayesian Multilevel, posterior inference'],
    ['4', 'SRP',             'Stacked ensemble: HLM, LASSO, KNN, RF, XGBoost'],
    ['5', 'MRdeeP',          'Wasserstein Generative Adversarial Network'],
]

col_widths = [0.05, 0.20, 0.75]

table = ax.table(
    cellText=data,
    colLabels=headers,
    colWidths=col_widths,
    loc='center',
    cellLoc='left',
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Style header
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)
    cell.set_edgecolor('white')

# Style body rows
for i in range(1, len(data) + 1):
    color = '#f7f9fc' if i % 2 == 0 else 'white'
    for j in range(len(headers)):
        cell = table[i, j]
        cell.set_facecolor(color)
        cell.set_edgecolor('#d5d8dc')
        if j == 1:
            cell.set_text_props(fontweight='bold')

# Center the # column
for i in range(len(data) + 1):
    table[i, 0].set_text_props(ha='center')

out_path = '/Users/carmenk/Documents/GitHub/MRdeeP-Deep-Learning-MRP/model_run_ces/model_specs_table.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01, facecolor='white')
print(f'Saved to {out_path}')
