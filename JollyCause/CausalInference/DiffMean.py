import pandas as pd
import statsmodels as st
import numpy as np

def ate_diff_mean(df,w,y):
    # Only valid in the randomized setting. Do not use in observational settings.
    Y = df[y]
    W = df[w]
    ate_est = np.mean(Y[W == 1]) - np.mean(Y[W == 0])
    ate_se = np.sqrt(np.var(Y[W == 1]) / sum(W == 1) + np.var(Y[W == 0]) / sum(W == 0))
    ate_tstat = ate_est / ate_se
    ate_pvalue = 2 * (1 - np.abs(ate_est / ate_se))
    ate_results = {"estimate": ate_est, "std.error": ate_se, "t.stat": ate_tstat, "pvalue": ate_pvalue}
    print(ate_results)
