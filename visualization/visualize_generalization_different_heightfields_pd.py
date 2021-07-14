import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import seaborn as sns

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


#from evaluation.compare_learning_performance_atEnd import boxplot_annotate_brackets_group

"""
    Visualizes generalization to novel, uneven terrain.
    
    All trained controller had been evaluated for 100 episodes,
    on different uneven terrains.
    Performance is measured and plotted here
    - as a figure showing how performance develops for the architectures 
    - boxplot for specific type of terrains
"""

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 
    
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

# Remove Type 3 fonts for latex
plt.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 3 - singe diag, 4 - single neig.
# 5 - two contr. diag, 6 - two neighb. contr.
# 7 - connections towards front
# Data was generated from compare_generalization_flat.py
path = '/home/nitro/ray_results/evaluation_1' # use your path
 # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

exp_name = ['Centralized', 'FullyDecentral', 'EightFullyDecentral', 'neighborJoint','neighborJoint2Legs','neighborJointAllInfo']
exp_name_written = ['Centralized', 'Four \n Fully \n Decentralized', 'Eight \n Fully \n Decentralized', 'Neigh. \n Joint', 'Neigh. \n Joint  \n 2  \n Legs', 'Neigh. \n Joint  \n All  \n Info']


# Plotting functions
####################
fig = plt.figure(figsize=(8, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 

#ax_arch.set_yscale('log')
ax_arch.set_xlim(1., 0.5)
ax_arch.set_xticks([1., 0.9, 0.8, 0.7, 0.6])
#ax_arch.set_ylim(0, 800)  

for i in range(0,len(exp_name)):
    # Use matplotlib's fill_between() call to create error bars.   
    #plt.fill_between(data_smoothn_steps, data_min[i,:],  
     #                data_max[i,:], color=tableau20[i*2 + 1], alpha=0.25)  
    mean_val = np.zeros(5)
    for j in range(0,5):
        mean_val[j] = np.mean(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and approach=="' + exp_name[i] + '"')['reward'])
    a, b = np.polyfit(data_smoothn_steps, mean_val, deg=1)
    # Provides a regression line and the inclination of that line:
    # 0.6  /  Centralized  - regression:  6389.723213619719
    # 0.6  /  FullyDecentral  - regression:  7574.152735948367
    # 0.6  /  Local  - regression:  7074.391503619364
    # 0.6  /  TwoSides  - regression:  8412.24418229771
    print(data_smoothn_steps[j], " / ", exp_name[i], " - regression: ", a)
    plt.plot(data_smoothn_steps, mean_val, color=tableau20[i*2], marker='+', lw=1, label=exp_name[i])

ax_arch.set_xlabel('Smoothness of evaluated terrain', fontsize=12)
ax_arch.set_ylabel('Mean return per Episode', fontsize=12)
plt.legend(loc="upper right")
#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')
file_name = "generalization_overSmoothness"
plt.savefig(file_name + '_legend.pdf')

#########################################
# Show Violin plot for evaluated runs
# on uneven terrain
#########################################
df_mean_eval_06 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

for appr_i in range(0,len(exp_name)):
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.6 and seed==' + str(seed_i) + \
            ' and approach=="' + exp_name[appr_i] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[appr_i], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_06 = df_mean_eval_06.append(new_mean_entry, ignore_index=True)

my_pal = {}
for i in range(0,len(exp_name)):
    my_pal[exp_name[i]] = tableau20[i*2]

#df_select = df.loc[df['Evaluate'].isin(['Flat','Height_010'])]
fig = plt.figure(figsize=(8, 6))
ax_mean_06 = plt.subplot(111) 
sns.set(style="ticks", color_codes=True)
sns.set(context="paper", palette="colorblind", style="ticks", font_scale=1.2)

#fig_violin = sns.FacetGrid(df_mean_eval_06, col="mean", sharey=True, size=4, aspect=.8)
fig_violin = sns.violinplot(ax=ax_mean_06, x="approach", y="mean", kind="violin", data=df_mean_eval_06, palette=my_pal, saturation=0.75)
fig_violin.spines['right'].set_visible(False)
fig_violin.spines['top'].set_visible(False)

ax_mean_06.set_xlabel('Comparison of architectures for uneven terrain (smoothness=0.6)', fontsize=12)
ax_mean_06.set_ylabel('Mean return for each seed', fontsize=12)

fig_violin.set_xticklabels(exp_name_written)


#########################################
# Box plot for evaluated runs
# on uneven terrain
#########################################

#df_select = df.loc[df['Evaluate'].isin(['Flat','Height_010'])]
fig_box = plt.figure(figsize=(8, 6))
ax_mean_box = plt.subplot(111) 
sns.set(style="ticks", color_codes=True)
sns.set(context="paper", palette="colorblind", style="ticks", font_scale=1.2)

#fig_violin = sns.FacetGrid(df_mean_eval_06, col="mean", sharey=True, size=4, aspect=.8)
fig_box = sns.boxplot(ax=ax_mean_box, x="approach", y="mean", data=df_mean_eval_06, palette=my_pal, saturation=0.75)
fig_box.spines['right'].set_visible(False)
fig_box.spines['top'].set_visible(False)

ax_mean_box.set_xlabel('Comparison of architectures for uneven terrain (smoothness=0.6)', fontsize=12)
ax_mean_box.set_ylabel('Mean return for each seed', fontsize=12)

fig_box.set_xticklabels(exp_name_written)

# Differences calculated in compare_generalization.pyplot
# We show two differences - these depend on the test here!
# sp.posthoc_mannwhitney(df_mean_eval_06, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# Centralized        1.000000        0.317232  1.000000             1.0        1.000000            1.0  1.000000  0.317232
# FullyDecentral     0.317232        1.000000  0.061662             1.0        0.128419            1.0  0.483209  1.000000
# Local              1.000000        0.061662  1.000000             1.0        1.000000            1.0  1.000000  0.036818
# SingleDiagonal     1.000000        1.000000  1.000000             1.0        1.000000            1.0  1.000000  1.000000
# SingleNeighbor     1.000000        0.128419  1.000000             1.0        1.000000            1.0  1.000000  0.101094
# SingleToFront      1.000000        1.000000  1.000000             1.0        1.000000            1.0  1.000000  1.000000
# TwoDiags           1.000000        0.483209  1.000000             1.0        1.000000            1.0  1.000000  0.483209
# TwoSides           0.317232        1.000000  0.036818             1.0        0.101094            1.0  0.483209  1.000000
# >>> sp.posthoc_dunn(df_mean_eval_06, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# Centralized        1.000000        0.085101  1.000000             1.0        1.000000            1.0  1.000000  0.085101
# FullyDecentral     0.085101        1.000000  0.012443             1.0        0.241236            1.0  0.336639  1.000000
# Local              1.000000        0.012443  1.000000             1.0        1.000000            1.0  1.000000  0.012443
# SingleDiagonal     1.000000        1.000000  1.000000             1.0        1.000000            1.0  1.000000  1.000000
# SingleNeighbor     1.000000        0.241236  1.000000             1.0        1.000000            1.0  1.000000  0.241236
# SingleToFront      1.000000        1.000000  1.000000             1.0        1.000000            1.0  1.000000  1.000000
# TwoDiags           1.000000        0.336639  1.000000             1.0        1.000000            1.0  1.000000  0.336639
# TwoSides           0.085101        1.000000  0.012443             1.0        0.241236            1.0  0.336639  1.000000

xpos_box = np.arange(0,len(exp_name))
heights_box = [np.max((df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[i]])["mean"]) for i in range(0,len(exp_name))]
#boxplot_annotate_brackets_group(1, [2], 'p < 0.05', xpos_box, heights_box)
#boxplot_annotate_brackets_group(2, [7], 'p < 0.05', xpos_box, heights_box, offset=100)
#boxplot_annotate_brackets_group(0, [1,2,3,4,5,7], 'p < 0.01', xpos_box, heights_box, offset=500)
fig.tight_layout()

#########################################
#########################################
# Box plot for evaluated runs
# on intermediate uneven terrain (Smoothness = 0.8)
#########################################
#########################################
df_mean_eval_08 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

for appr_i in range(0,len(exp_name)):
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.8 and seed==' + str(seed_i) + \
            ' and approach=="' + exp_name[appr_i] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[appr_i], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_08 = df_mean_eval_08.append(new_mean_entry, ignore_index=True)

# Show distributions for smoothness 0.8
#########################################

#df_select = df.loc[df['Evaluate'].isin(['Flat','Height_010'])]
fig = plt.figure(figsize=(8, 6))
ax_mean_box_08 = plt.subplot(111) 
sns.set(style="ticks", color_codes=True)
sns.set(context="paper", palette="colorblind", style="ticks", font_scale=1.2)

#fig_violin = sns.FacetGrid(df_mean_eval_08, col="mean", sharey=True, size=4, aspect=.8)
fig_box_08 = sns.boxplot(ax=ax_mean_box_08, x="approach", y="mean", data=df_mean_eval_08, palette=my_pal, saturation=0.75)
fig_box_08.spines['right'].set_visible(False)
fig_box_08.spines['top'].set_visible(False)

ax_mean_box_08.set_xlabel('Comparison of architectures for uneven terrain (smoothness=0.8)', fontsize=12)
ax_mean_box_08.set_ylabel('Mean return for each seed', fontsize=12)

fig_box_08.set_xticklabels(exp_name_written)

# Differences calculated in compare_generalization.pyplot
# Shown here (only significant row kept)
# sp.posthoc_mannwhitney(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# FullyDecentral     1.000000        1.000000  0.009235        1.000000        0.006892       0.255038       1.0  0.101094

# sp.posthoc_dunn(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# FullyDecentral     1.000000        1.000000  0.001068        1.000000        0.008951            1.0  1.000000  0.515138

xpos_box = np.arange(0,len(exp_name))
heights_box = [np.max((df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[i]])["mean"]) for i in range(0,len(exp_name))]
#boxplot_annotate_brackets_group(1, [2,4], 'p < 0.01', xpos_box, heights_box)
#boxplot_annotate_brackets_group(7, [6,3], 'p < 0.05', xpos_box, heights_box, offset=300)
#boxplot_annotate_brackets_group(0, [1,2,3,4,5,7], 'p < 0.01', xpos_box, heights_box, offset=500)
fig.tight_layout()

#df_select = df.loc[df['Evaluate'].isin(['Flat','Height_010'])]
fig_08 = plt.figure(figsize=(8, 6))
ax_mean_08 = plt.subplot(111) 
sns.set(style="ticks", color_codes=True)
sns.set(context="paper", palette="colorblind", style="ticks", font_scale=1.2)

#fig_violin = sns.FacetGrid(df_mean_eval_08, col="mean", sharey=True, size=4, aspect=.8)
fig_violin_08 = sns.violinplot(ax=ax_mean_08, x="approach", y="mean", kind="violin", data=df_mean_eval_08, palette=my_pal, saturation=0.75)
fig_violin_08.spines['right'].set_visible(False)
fig_violin_08.spines['top'].set_visible(False)

ax_mean_08.set_xlabel('Comparison of architectures for uneven terrain (smoothness=0.8)', fontsize=12)
ax_mean_08.set_ylabel('Mean return for each seed', fontsize=12)

fig_violin_08.set_xticklabels(exp_name_written)

plt.show()
