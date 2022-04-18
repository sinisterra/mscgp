def enforce_restrictions(evaluation):
    return (
        True
        and evaluation["tp"] >= 1
        # and evaluation["absolute_risk"] > 0
        #         and (evaluation["has_redundant_selectors"] == False)
        # and evaluation["is_n_closed"]
        # and evaluation["support"] > 0
        # and evaluation["significant"]
        # and evaluation["odds"] >= 1
        # and evaluation["lift"] >= 1
        # and evaluation["min_lift"]
        # and evaluation["significant"]
        # and not evaluation["paradoxical"]
        # and evaluation["certainty"] < 1
        # and evaluation["tp"] > evaluation["fp"]
        # and evaluation["tn"] > evaluation["fn"]
        # and evaluation["mcc"] >= 0.2
        # and evaluation["tp"] > evaluation["fp"]
        # and evaluation["fpr"] <= evaluation["prevalence"]
        # and evaluation["mcc"] >= 0.3
        #        and evaluation["certainty"] >= 0.5
        # and evaluation["markedness"] > 0.1
        # and evaluation["confidence"] < 1
        # and evaluation["cer"] < 0.8
        # and evaluation["cer"] > 0.2
        # and evaluation["support"] > 0
        # and evaluation["absolute_risk"] > 0
        # and evaluation["fn"] > 1
        # and evaluation["fp"] > 1
        # and evaluation["tp"] > 1
        # and evaluation["tn"] > 1
        # and evaluation["absolute_risk"] < 0.8
        # and evaluation["prevalence"] > 0.1
    )
