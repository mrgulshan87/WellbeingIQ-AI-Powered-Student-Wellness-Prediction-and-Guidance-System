"""Explainability helper for WellbeingIQ: SHAP contributions grouped into readable factors.
Used by Notebook 4, the GenAI wellness coach and the dashboard."""
import json, numpy as np, pandas as pd, joblib, shap

LABELS = {
 "Age":"Age","Gender":"Gender","Country":"Country","Academic_Level":"Academic level","Residence":"Residence",
 "Most_Used_Platform":"Most used platform","Purpose_Of_Use":"Purpose of use",
 "Avg_Daily_Usage_Hours":"Daily social media use","Total_Screen_Time_Hours":"Total daily screen time",
 "Gaming_Hours":"Gaming hours","Daily_Unlocks":"Phone unlocks per day","Notifications_Per_Day":"Notifications per day",
 "Late_Night_Usage_Hours":"Late-night phone use","Study_Hours":"Study hours","Sleep_Hours_Per_Night":"Sleep per night",
 "Sleep_Quality":"Sleep quality","Caffeine_Cups_Per_Day":"Caffeine per day","Physical_Activity_Hours":"Physical activity",
 "Days_Offline_Per_Month":"Days offline per month","Social_Support_Score":"Social support","Academic_Pressure":"Academic pressure",
 "Academic_Performance_Pct":"Academic performance","Conflicts_Over_Social_Media":"Conflicts over social media use",
 "Addiction_Score":"Self-rated addiction level","Usage_Sleep_Combination":"Heavy use combined with short sleep"}
# engineered features are merged into the habit they describe, so explanations stay readable
UNITS = {"Avg_Daily_Usage_Hours":" h","Late_Night_Usage_Hours":" h","Sleep_Hours_Per_Night":" h","Physical_Activity_Hours":" h","Study_Hours":" h",
         "Gaming_Hours":" h","Total_Screen_Time_Hours":" h","Social_Support_Score":"/10","Academic_Pressure":"/10","Sleep_Quality":"/5",
         "Conflicts_Over_Social_Media":"/5","Caffeine_Cups_Per_Day":" cups","Days_Offline_Per_Month":" days","Academic_Performance_Pct":" percent"}
ENGINEERED_GROUP = {"Sleep_Deficit":"Sleep_Hours_Per_Night","Late_Night_Share":"Late_Night_Usage_Hours",
                    "Usage_Sleep_Ratio":"Usage_Sleep_Combination","HighUsage_LowSleep":"Usage_Sleep_Combination"}

def group_of(transformed_name, cat_cols):
    n = transformed_name
    if n.startswith("cat__"):
        for c in cat_cols:
            if n.startswith("cat__" + c + "_"): return c
    n = n.replace("num__", "")
    return ENGINEERED_GROUP.get(n, n)

class WellbeingExplainer:
    def __init__(self, model_path="wellbeingiq_model.joblib", train_csv="X_train.csv",
                 card_path="model_card.json", config_path="explain_config.json"):
        self.pipe = joblib.load(model_path)
        self.features = json.load(open(card_path))["features"]
        self.fe, self.prep, self.clf = (self.pipe.named_steps[k] for k in ("fe", "prep", "clf"))
        self.cat_cols = list(self.prep.transformers_[1][2])
        self.names = list(self.prep.get_feature_names_out())
        self.groups = [group_of(n, self.cat_cols) for n in self.names]
        self.classes = list(self.clf.classes_)
        bg = pd.read_csv(train_csv)[self.features]
        bgt = self.prep.transform(self.fe.transform(bg))
        self.explainer = shap.LinearExplainer(self.clf, shap.maskers.Independent(bgt, max_samples=len(bgt)))
        try: self.uncertain_below = json.load(open(config_path))["uncertainty_confidence_threshold"]
        except Exception: self.uncertain_below = 0.5

    def transform(self, X):
        return self.prep.transform(self.fe.transform(X[self.features]))

    def shap_values(self, Xt):
        sv = self.explainer.shap_values(Xt)
        return np.stack(sv, axis=-1) if isinstance(sv, list) else sv      # (n, features, classes)

    def grouped(self, sv_2d):
        """sum SHAP columns that belong to the same readable factor -> DataFrame (n, groups)"""
        return pd.DataFrame(sv_2d, columns=self.groups).T.groupby(level=0).sum().T

    def _value_text(self, group, raw_row, fe_row):
        if group == "Usage_Sleep_Combination":
            return f"{raw_row['Avg_Daily_Usage_Hours']:.1f} h use, {raw_row['Sleep_Hours_Per_Night']:.1f} h sleep"
        v = raw_row.get(group, np.nan)
        return "missing" if pd.isna(v) else (f"{float(v):.1f}".rstrip("0").rstrip(".") + UNITS.get(group, "") if isinstance(v, (int, float, np.number)) else v)

    def explain(self, row, top_k=3):
        """row: dict or one-row DataFrame with the model features. Returns a plain dict (JSON friendly)."""
        X = pd.DataFrame([row]) if isinstance(row, dict) else row.reset_index(drop=True)
        Xt = self.transform(X); proba = self.pipe.predict_proba(X[self.features])[0]
        ai = self.classes.index("At-Risk")
        g = self.grouped(self.shap_values(Xt)[:, :, ai]).iloc[0].sort_values()
        raw = X.iloc[0]
        mk = lambda s: [{"feature": k, "label": LABELS.get(k, k), "value": self._value_text(k, raw, None), "impact": round(float(v), 3)} for k, v in s.items()]
        drivers = mk(g[g > 0].sort_values(ascending=False).head(top_k)); protect = mk(g[g < 0].head(top_k))
        conf = float(proba.max())
        return {"predicted_category": self.classes[int(proba.argmax())],
                "probabilities": {c: round(float(p), 3) for c, p in zip(self.classes, proba)},
                "confidence": round(conf, 3), "uncertain": conf < self.uncertain_below,
                "risk_drivers": drivers, "protective_factors": protect,
                "note": "impact = contribution to At-Risk log-odds (positive raises risk, negative lowers it)"}
