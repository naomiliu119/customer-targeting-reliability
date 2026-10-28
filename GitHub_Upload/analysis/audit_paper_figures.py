"""Independently refit the original models and check Figure 1 and audience counts.

Usage: python analysis/audit_paper_figures.py /path/to/hillstrom.csv
Requires the original notebook's NumPy, pandas, SciPy and scikit-learn dependencies.
This checks saved results; it does not alter model settings or targeting rules.
"""

from pathlib import Path
from statistics import NormalDist
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def preprocess():
    return ColumnTransformer([
        ("numeric", StandardScaler(), ["recency", "history", "mens", "womens", "newbie"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["channel", "zip_code"]),
    ])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    args=parser.parse_args()
    sha=hashlib.sha256(args.data.read_bytes()).hexdigest()
    assert sha=="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
    raw=pd.read_csv(args.data)
    frame=raw[raw.segment.isin(["Mens E-Mail","No E-Mail"])].copy()
    frame["treatment"]=(frame.segment=="Mens E-Mail").astype(int)
    train,rem=train_test_split(frame,test_size=.4,stratify=frame.treatment,random_state=42)
    valid,test=train_test_split(rem,test_size=.5,stratify=rem.treatment,random_state=42)
    emailed=train[train.treatment==1]
    control=train[train.treatment==0]
    purchase=Pipeline([("preprocess",preprocess()),("model",LogisticRegression(max_iter=2000))])
    revenue=Pipeline([("preprocess",preprocess()),("model",Ridge(alpha=1.0))])
    no_email=Pipeline([("preprocess",preprocess()),("model",Ridge(alpha=1.0))])
    purchase.fit(emailed,emailed.conversion)
    revenue.fit(emailed,emailed.spend)
    no_email.fit(control,control.spend)
    scores={"Historical spending":test.history.to_numpy(),
            "Purchase probability":purchase.predict_proba(test)[:,1],
            "Predicted revenue":revenue.predict(test),
            "Incremental revenue":revenue.predict(test)-no_email.predict(test)}
    n=len(test); k=int(.25*n); q=k/n
    z=2*test.treatment.to_numpy()*test.spend.to_numpy()-2*(1-test.treatment.to_numpy())*test.spend.to_numpy()
    original=pd.read_csv(ROOT/"analysis/results/original_notebook/final_test_comparisons.csv")
    original=original[original["Contact rate"]=="25%"].set_index("Rule")
    selections={};rows=[]
    for name,score in scores.items():
        selected=pd.Series(False,index=test.index)
        selected.loc[pd.Series(score,index=test.index).nlargest(k,keep="first").index]=True
        selections[name]=selected.to_numpy()
        contributions=1000*(selected.to_numpy().astype(float)-q)*z
        mean=float(contributions.mean())
        se=float(contributions.std(ddof=1)/np.sqrt(n))
        critical=NormalDist().inv_cdf(.975)
        values={"Additional revenue":float(1000*np.mean(selected.to_numpy()*z)),
                "Advantage over random":mean,"95% CI lower":mean-critical*se,"95% CI upper":mean+critical*se}
        for column,value in values.items():
            assert np.isclose(value,original.loc[name,column],rtol=1e-9,atol=1e-8),(name,column,value,original.loc[name,column])
        rows.append({"rule":name,**values})
    shared=int((selections["Historical spending"] & selections["Incremental revenue"]).sum())
    assert shared==1288 and k-shared==842
    frequencies=pd.read_csv(ROOT/"analysis/results/selection_frequencies.csv").set_index("source_row")
    stored=frequencies.loc[test.index,"original_selected"].to_numpy(bool)
    assert np.array_equal(stored,selections["Incremental revenue"])
    observed=frequencies.loc[frequencies.original_selected,"selection_frequency"].to_numpy()
    assert len(observed)==2130 and int((observed<.5).sum())==524
    assert np.isclose(np.histogram(observed,bins=np.linspace(0,1,21))[0].sum(),2130)
    ev=valid[valid.treatment==1]
    validation={"purchase_roc_auc":float(roc_auc_score(ev.conversion,purchase.predict_proba(ev)[:,1])),
                "purchase_log_loss":float(log_loss(ev.conversion,purchase.predict_proba(ev)[:,1]))}
    assert round(validation["purchase_roc_auc"],4)==.5853
    assert round(validation["purchase_log_loss"],4)==.0751
    result={"dataset_sha256":sha,"versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
            "figure_1_refitted_model_results":rows,
            "figure_2":{"selected_per_rule":k,"shared":shared,"unique_per_rule":k-shared,"original_selected_below_half":int((observed<.5).sum()),"histogram_total":len(observed)},
            "purchase_validation_independently_refitted":validation,
            "scope":"Original model fits and Figure 1 normal intervals independently reproduced; Figure 2 membership checked. Training refits were separately rerun and compared in this review. Original percentile bootstrap intervals were not regenerated by this script."}
    output=ROOT/"analysis/results/figure_audit.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
