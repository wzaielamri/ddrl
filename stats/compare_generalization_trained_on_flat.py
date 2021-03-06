import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

from scipy import stats
import scikit_posthocs as sp

"""
    Calculate statistics from evaluation runs for experiment one
    (trained on flat terrain, aiming for maximal velocity).
    
    Eight architectures are used for training and evaluation.
    
    Script groups results from evaluation (for each arch. 10 seeds, 100 runs), 
    calculates mean values for return on different terrains.
    
    This produces statistics as given in Sect. 4.2
    
    Input: 
        Evaluation runs for controller (see sect. 4.2)
        1_trained_flat_eval - trained on flat terrain
            output of evaluate script: evaluate_trained_policies_pd.py
        1_trained_hf06_eval - trained on uneven terrain (smooothness 0.6, S-III.)
            here used for comparison to expert policies: trained already on uneven terrain
"""

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']
# Load evaluation results from first experiment 1, sect. 4.2
path = os.getcwd() + '/Results/1_trained_flat_eval' # use your path
select_arch = [0,1,2,3,4,5,6,7]
# Load for comparison evaluation results for trained experts (on height field 
# of smoothness 0.6, supplement S-III.)
#path = os.getcwd() + '/Results/1_trained_hf06_eval' # use your path
#select_arch = [0,1,2,7]
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

df_mean_eval_06 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

for appr_i in select_arch:
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.6 and seed==' + str(seed_i) + \
            ' and approach=="' + exp_name[appr_i] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[appr_i], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_06 = df_mean_eval_06.append(new_mean_entry, ignore_index=True)

df_mean_eval_08 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

for appr_i in select_arch:
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.8 and seed==' + str(seed_i) + \
            ' and approach=="' + exp_name[appr_i] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[appr_i], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_08 = df_mean_eval_08.append(new_mean_entry, ignore_index=True)

#########################################
# Statistics: Test for differences between groups
#
# We analyzed the performance (mean return) of the unpaired samples from the different 
# architectures using the non-parametric Kruskal-Wallis test [Kruskal & Wallis 1952] 
# (as the data appears not normally distributed) and for post-hoc analysis using the 
# Dunn [Dunn & Dunn 1961] post-hoc test (applying Bonferroni correction) 
# following [Raff 2019].
#
# We compared the mean return on two different, novel terrains.
#########################################

# For smoothness 0.6
####################
stats.kruskal( (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[0]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[1]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[2]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[3]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[4]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[5]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[6]])["mean"],
    (df_mean_eval_06.loc[df_mean_eval_06["approach"] == exp_name[7]])["mean"] )
    
#KruskalResult(statistic=23.732592592592596, pvalue=0.0012694133089161625)
# Eta square 0.2324
# Epsilon Squared = H / ((n^2-1)/(n+1)); 0.300
# = relatively strong
sp.posthoc_dunn(df_mean_eval_06, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
# OR posthoc_mannwhitney, posthoc_conover

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

# For smoothness 0.8
####################
stats.kruskal( (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[0]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[1]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[2]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[3]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[4]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[5]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[6]])["mean"],
    (df_mean_eval_08.loc[df_mean_eval_08["approach"] == exp_name[7]])["mean"] )
    
#KruskalResult(statistic=22.443703703703704, pvalue=0.0021292970018814317)
# k - number of groups, n - total number of observations
# Eta square = (H - k + 1)/ (n - k) = 0.2145 - for partial eta square
# = large effect
# 0.01- < 0.06 (small effect), 0.06 - < 0.14 (moderate effect) and >= 0.14 (large effect).
# https://rpkgs.datanovia.com/rstatix/reference/kruskal_effsize.html
# Epsilon Squared = H / ((n^2-1)/(n+1)); 0.2841
# = relatively strong
# Rea & Parker (1992) their interpretation for r, the following:
#0.00 < 0.01 - Negligible
#0.01 < 0.04 - Weak
#0.04 < 0.16 - Moderate
#0.16 < 0.36 - Relatively strong
#0.36 < 0.64 - Strong
#0.64 < 1.00 - Very strong
#sp.posthoc_mannwhitney(architecture_samples_at312, p_adjust = 'holm')
#A Kruskal-Wallis test showed that Location had a significant relatively strong effect on how motivated students were by the teacher, χ2(2, N = 54) = 21.33, p < .001, ε2 = .40. A post-hoc test using Dunn's test with Bonferroni correction showed the significant differences between Diemen and Haarlem, p < .05, and between Diemen and Rotterdam, p < .001
sp.posthoc_dunn(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
# OR posthoc_mannwhitney, posthoc_conover

# Outputs are shown for results from Experiment 1 (trained on flat terrain).
# sp.posthoc_mannwhitney(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# Centralized        1.000000        1.000000  0.873852        1.000000        1.000000       1.000000       1.0  1.000000
# FullyDecentral     1.000000        1.000000  0.009235        1.000000        0.006892       0.255038       1.0  0.101094
# Local              0.873852        0.009235  1.000000        0.720946        1.000000       0.873852       1.0  1.000000
# SingleDiagonal     1.000000        1.000000  0.720946        1.000000        1.000000       1.000000       1.0  1.000000
# SingleNeighbor     1.000000        0.006892  1.000000        1.000000        1.000000       1.000000       1.0  1.000000
# SingleToFront      1.000000        0.255038  0.873852        1.000000        1.000000       1.000000       1.0  1.000000
# TwoDiags           1.000000        1.000000  1.000000        1.000000        1.000000       1.000000       1.0  1.000000
# TwoSides           1.000000        0.101094  1.000000        1.000000        1.000000       1.000000       1.0  1.000000
# 
# sp.posthoc_dunn(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# Centralized        1.000000        1.000000  0.215289        1.000000        0.892849            1.0  1.000000  1.000000
# FullyDecentral     1.000000        1.000000  0.001068        1.000000        0.008951            1.0  1.000000  0.515138
# Local              0.215289        0.001068  1.000000        0.850719        1.000000            1.0  0.734423  1.000000
# SingleDiagonal     1.000000        1.000000  0.850719        1.000000        1.000000            1.0  1.000000  1.000000
# SingleNeighbor     0.892849        0.008951  1.000000        1.000000        1.000000            1.0  1.000000  1.000000
# SingleToFront      1.000000        1.000000  1.000000        1.000000        1.000000            1.0  1.000000  1.000000
# TwoDiags           1.000000        1.000000  0.734423        1.000000        1.000000            1.0  1.000000  1.000000
# TwoSides           1.000000        0.515138  1.000000        1.000000        1.000000            1.0  1.000000  1.000000
# 
# sp.posthoc_conover(df_mean_eval_08, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#             Centralized  FullyDecentral     Local  SingleDiagonal  SingleNeighbor  SingleToFront  TwoDiags  TwoSides
# Centralized        1.000000        1.000000  0.101562        1.000000        0.503833       1.000000  1.000000  1.000000
# FullyDecentral     1.000000        1.000000  0.000416        0.860240        0.003445       0.577692  1.000000  0.269538
# Local              0.101562        0.000416  1.000000        0.476736        1.000000       0.716011  0.403105  1.000000
# SingleDiagonal     1.000000        0.860240  0.476736        1.000000        1.000000       1.000000  1.000000  1.000000
# SingleNeighbor     0.503833        0.003445  1.000000        1.000000        1.000000       1.000000  1.000000  1.000000
# SingleToFront      1.000000        0.577692  0.716011        1.000000        1.000000       1.000000  1.000000  1.000000
# TwoDiags           1.000000        1.000000  0.403105        1.000000        1.000000       1.000000  1.000000  1.000000
# TwoSides           1.000000        0.269538  1.000000        1.000000        1.000000       1.000000  1.000000  1.000000

plt.show()
