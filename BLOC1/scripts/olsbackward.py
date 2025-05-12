def olsbackward(data, start, crit="aic", verbose=False):
    """Backward selection for linear model with smf (with formula).

    Parameters:
    -----------
    data (pandas DataFrame): DataFrame with all possible predictors
            and response
    start (string): a string giving the starting model
            (ie the starting point)
    crit (string): "aic"/"AIC" or "bic"/"BIC"
    verbose (boolean): if True verbose print

    Returns:
    --------
    model: an "optimal" linear model fitted with statsmodels
           with an intercept and
           selected by forward/backward or both algorithm with crit criterion
    """
    # criterion
    if not (crit == "aic" or crit == "AIC" or crit == "bic" or crit == "BIC"):
        raise ValueError("criterion error (should be AIC/aic or BIC/bic)")
    # starting point
    formula_start = start.split("~")
    response = formula_start[0].strip()
    # explanatory variables for the 3 models
    start_explanatory = set([item.strip() for item in
                             formula_start[1].split("+")]) - set(['1'])
    # setting up the set "remove" which contains the possible
    # variable to remove
    lower_explanatory = set([])
    remove = start_explanatory - lower_explanatory
    # current point
    selected = start_explanatory
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(list(selected)))
    if crit == "aic" or crit == "AIC":
        current_score = smf.ols(formula, data).fit().aic
    elif crit == "bic" or crit == "BIC":
        current_score = smf.ols(formula, data).fit().bic
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Starting", selected))
    # main loop
    while True:
        scores_with_candidates = []
        for candidate in remove:
            tobetested = selected - set([candidate])
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(list(tobetested)))
            if crit == "aic" or crit == "AIC":
                score = smf.ols(formula, data).fit().aic
            elif crit == "bic" or crit == "BIC":
                score = smf.ols(formula, data).fit().bic
            if verbose:
                print((score, "-", candidate))
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            remove = remove - set([best_candidate])
            selected = selected - set([best_candidate])
            current_score = best_new_score
            if verbose:
                print("----------------------------------------------")
                print((current_score, "New Current", selected))
        else:
            break
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Final", selected))
    formula = "{} ~ {} + 1".format(response, ' + '.join(list(selected)))
    model = smf.ols(formula, data).fit()
    return model