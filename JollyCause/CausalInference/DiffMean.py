import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy

def ate_diff_mean(df,w,y):
    # Only valid in the randomized setting. Do not use in observational settings.
    Y = df[y]
    W = df[w]
    ate_est = np.mean(Y[W == 1]) - np.mean(Y[W == 0])
    ate_se = np.sqrt(np.var(Y[W == 1]) / sum(W == 1) + np.var(Y[W == 0]) / sum(W == 0))
    ate_tstat = ate_est / ate_se
    ate_pvalue = 2 * (scipy.stats.norm.cdf(1 - np.abs(ate_est / ate_se)))
    ate_results = {"estimate": ate_est, "std.error": ate_se, "t.stat": ate_tstat, "pvalue": ate_pvalue}
    print(ate_results)

def ate_ls_model(df,w,y,hc=True):

    formula = "y ~ w"

    # Fit the OLS model
    if hc:
        ols = sm.OLS.from_formula(formula, data=df).fit(cov_type='HC2')
    else:
        ols = sm.OLS.from_formula(formula, data=df).fit()
    #ols = sm.OLS(outcome,treatment).fit()

    # Extract coefficient summary for the treatment variable
    coef_summary = ols.summary().tables[1]
    coef_summary = pd.DataFrame(coef_summary.data[1:], columns=coef_summary.data[0])
    coef_treatment = pd.DataFrame(coef_summary.iloc[1:])
    # Extract the coefficient, standard error, t-value, and p-value
    coef_estimate = float(coef_treatment['coef'])
    std_error = float(coef_treatment['std err'])
    if hc:
        t_value = float(coef_treatment['z'])
        p_value = float(coef_treatment['P>|z|'])
    else:
        t_value = float(coef_treatment['t'])
        p_value = float(coef_treatment['P>|t|'])

    ols_results = {
    'estimate': coef_estimate,
    'std_error': std_error,
    't_value': t_value,
    'p_value': p_value
    }

    print(ols_results)

def propensity_score(df,treatment:str,covariates:list,inplace=False):

    # Create a formula for logistic regression
    formula = f"{treatment} ~ {' + '.join(covariates)}"

    # Fit logistic regression model
    logit_model = sm.Logit.from_formula(formula, data=df)
    logit_result = logit_model.fit()

    # Extract propensity scores
    Propensity_score=logit_result.predict(df)
    if inplace:
        df['propensity_score'] = Propensity_score

    # Display the DataFrame with propensity scores
    return Propensity_score

def IPW(df,treatment:str,covariates:list):
    df1=df.copy()
    df1['propensity_score']=propensity_score(df1,treatment,covariates)
    treatment_group = df1[df1[treatment] == 1]
    control_group = df1[df1[treatment] == 0]

    # Calculate weights
    treatment_weights = 1 / treatment_group['propensity_score']
    control_weights = 1 / (1 - control_group['propensity_score'])

    # Combine treatment and control weights
    df1['weights'] = pd.concat([treatment_weights, control_weights])

    # Calculate inverse propensity-weighted estimator
    df1['weighted_outcome'] = df1['y'] * df1['weights']

    # Calculate the average treatment effect
    ate_est = df1['weighted_outcome'].mean()


    weighted_ols = sm.WLS(df1['y'], sm.add_constant(df1[treatment]), weights=df1['weights']).fit()
    coef_estimate=pd.DataFrame(ols.conf_int()).iloc[1,0]
    std_error = weighted_ols.bse[treatment]
    t_stat = weighted_ols.tvalues[treatment]
    p_value = weighted_ols.pvalues[treatment]

    IPW_results = {
    'estimate': coef_estimate,
    'std_error': std_error,
    't_value': t_value,
    'p_value': p_value
    }
    print(IPW_results)



def AIPW(df,treatment:str,covariates:list):
    df1=df.copy()
    df1['propensity_score']=propensity_score(df1,treatment,covariates)
    treatment_group = df1[df1[treatment] == 1]
    control_group = df1[df1[treatment] == 0]

    # Calculate weights
    treatment_weights = 1 / treatment_group['propensity_score']
    control_weights = 1 / (1 - control_group['propensity_score'])

    # Combine treatment and control weights
    df1['weights'] = pd.concat([treatment_weights, control_weights])

    # Calculate inverse propensity-weighted estimator
    X_augmented = sm.add_constant(df1[covariates + [treatment]])

    # Calculate augmented inverse propensity-weighted estimator
    weighted_ols = sm.WLS(df1['y'], X_augmented, weights=df1['weights']).fit()

    # Extract results
    ate_est = weighted_ols.params[treatment]
    std_error = weighted_ols.bse[treatment]
    t_stat = weighted_ols.tvalues[treatment]
    p_value = weighted_ols.pvalues[treatment]

    AIPW_results = {
        'estimate': ate_est,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value
    }

    print(AIPW_results)


df = pd.read_csv("https://docs.google.com/uc?id=1AQva5-vDlgBcM_Tv9yrO8yMYRfQJgqo_&export=download")
#ate_ls_model(df,'w','y')
#ate_ls_model(df,'w','y',hc=False)
#ate_diff_mean(df,'w','y')
#propensity_score(df,'w',['educ','polviews','age'])
#IPW(df,'w',['educ','polviews','age'])
AIPW(df,'w',['educ','polviews','age'])
