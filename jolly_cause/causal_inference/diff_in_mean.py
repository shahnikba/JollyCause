''' import libraries'''
import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy.stats
from sklearn.neighbors import NearestNeighbors


def ate_diff_mean(data, treatment, outcome):
    '''Only valid in the randomized setting. Do not use in observational settings.'''
    outcome = data[outcome]
    treatment = data[treatment]
    treated_group=outcome[treatment == 1]
    control_group=outcome[treatment == 0]
    ate_est = np.mean(treated_group) - np.mean(control_group)
    ate_se =np.sqrt(np.var(treated_group)/sum(treatment==1)+np.var(control_group)/sum(treatment==0))
    ate_tstat = ate_est / ate_se
    ate_pvalue = 2 * (scipy.stats.norm.cdf(1 - np.abs(ate_est / ate_se)))
    ate_results =  {
        "estimate": ate_est,
        "std.error": ate_se,
        "t.stat": ate_tstat,
        "pvalue": ate_pvalue
    }
    print(ate_results)

def ate_ls_model(data, treatment, outcome, hc_test=True):
    '''average treatment efect by lienar regression'''
    formula = f"{outcome} ~ {treatment}"

    # Fit the OLS model
    if hc_test:
        ols = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC2')
    else:
        ols = sm.OLS.from_formula(formula, data=data).fit()

    # Extract coefficient summary for the treatment variable
    coef_summary = ols.summary().tables[1]
    coef_summary = pd.DataFrame(coef_summary.data[1:], columns=coef_summary.data[0])
    coef_treatment = pd.DataFrame(coef_summary.iloc[1:])
    # Extract the coefficient, standard error, t-value, and p-value
    coef_estimate = float(coef_treatment['coef'])
    std_error = float(coef_treatment['std err'])
    if hc_test:
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

def propensity_score(data, treatment: str, covariates: list, inplace=False):
    '''Create a formula for logistic regression'''
    formula = f"{treatment} ~ {' + '.join(covariates)}"

    # Fit logistic regression model
    logit_model = sm.Logit.from_formula(formula, data=data)
    logit_result = logit_model.fit()

    # Extract propensity scores
    p_score = logit_result.predict(data)
    if inplace:
        data['propensity_score'] = p_score

    # Display the DataFrame with propensity scores
    return p_score

def ipw(data, outcome:str ,treatment: str, covariates: list):
    '''inverce propensity score'''
    data1 = data.copy()
    data1['propensity_score'] = propensity_score(data1, treatment, covariates)
    treatment_group = data1[data1[treatment] == 1]
    control_group = data1[data1[treatment] == 0]

    # Calculate weights
    treatment_weights = 1 / treatment_group['propensity_score']
    control_weights = 1 / (1 - control_group['propensity_score'])

    # Combine treatment and control weights
    data1['weights'] = pd.concat([treatment_weights, control_weights])

    # Calculate inverse propensity-weighted estimator
    data1['weighted_outcome'] = data1[outcome] * data1['weights']

    # Calculate the average treatment effect
    ate_est = data1['weighted_outcome'].mean()
    constent=sm.add_constant(data1[treatment])
    weighted_ols = sm.WLS(data1[outcome], constent, weights=data1['weights']).fit()
    coef_estimate = pd.DataFrame(weighted_ols.conf_int()).iloc[1, 0]

    ipw_results = {
        'ate_est':ate_est,
        'estimate': coef_estimate,
        'std_error': weighted_ols.bse[treatment],
        't_value': weighted_ols.tvalues[treatment],
        'p_value': weighted_ols.pvalues[treatment]
    }
    print(ipw_results)

def aipw(data, outcome:str ,treatment: str, covariates: list):
    '''agmented inverce propensity score'''
    data1 = data.copy()
    data1['propensity_score'] = propensity_score(data1, treatment, covariates)
    treatment_group = data1[data1[treatment] == 1]
    control_group = data1[data1[treatment] == 0]

    # Calculate weights
    treatment_weights = 1 / treatment_group['propensity_score']
    control_weights = 1 / (1 - control_group['propensity_score'])
    # Combine treatment and control weights
    data1['weights'] = pd.concat([treatment_weights, control_weights])

    # Calculate inverse propensity-weighted estimator
    x_augmented = sm.add_constant(data1[covariates + [treatment]])

    # Calculate augmented inverse propensity-weighted estimator
    weighted_ols = sm.WLS(data1[outcome], x_augmented, weights=data1['weights']).fit()

    # Extract results
    aipw_results = {
        'estimate': weighted_ols.params[treatment],
        'std_error': weighted_ols.bse[treatment],
        't_stat': weighted_ols.tvalues[treatment],
        'p_value': weighted_ols.pvalues[treatment]
    }

    print(aipw_results)


#df = pd.read_csv("https://docs.google.com/uc?id=1AQva5-vDlgBcM_Tv9yrO8yMYRfQJgqo_&export=download")
#ate_ls_model(df,'w','y')
#ate_ls_model(df,'w','y',hc_test=False)
#ate_diff_mean(df,'w','y')
#propensity_score(df,'w',['educ','polviews','age'])
#ipw(df,'y','w',['age'])
#aipw(df,'y','w',['educ','polviews','age'])




def propensity_score_matching(data, treatment_col, covariates):
    """
    Perform propensity score matching.

    Parameters:
    - data: DataFrame containing the data
    - treatment_col: Name of the column indicating treatment/control status
    - covariates: List of covariates used for propensity score calculation
    - caliper: Caliper width for matching (default is 0.05)

    Returns:
    - matched_data: DataFrame containing the matched data
    """

    # Calculate propensity scores
    propensity_scores = propensity_score(data, treatment_col, covariates)

    # Add propensity scores to the original DataFrame
    data['propensity_score'] = propensity_scores

    # Separate treated and control groups
    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]

    # Initialize matched data
    matched_units = []


    knn_model = NearestNeighbors(n_neighbors=1)
    knn_model.fit(control_data[['propensity_score']])

    # Find nearest neighbors for each treated unit
    _, indices = knn_model.kneighbors(treated_data[['propensity_score']])

    # Initialize matched data

    # Iterate through each treated unit and its nearest neighbor(s)
    for i, treated_unit in enumerate(treated_data.iterrows()):
        treated_unit = treated_unit[1]
        nearest_indices = indices[i]

        # Add the treated unit and its nearest neighbor(s) to the result
        matched_units.append(treated_unit)
        matched_units.append(control_data.iloc[nearest_indices].squeeze())

    matched_data = pd.DataFrame(matched_units)

    return matched_data
