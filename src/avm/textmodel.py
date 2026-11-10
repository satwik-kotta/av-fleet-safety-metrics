"""Text model for disengagement causes, its honest evaluation, and similarity search.

Training labels come from the rules in labels.py, so they are noisy. The point here is
whether a model trained on some companies' wording can label causes written by a
company it has never seen, and how much a naive evaluation overstates that.
"""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize

SEED = 7


def _model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0),
    )


def evaluate(ev: pd.DataFrame) -> dict:
    """ev needs columns company, cause_text, template, cat, amb."""
    lab = ev[(ev["cat"] != "Unclear") & ~ev["amb"]]

    # 1. Naive: random split of event rows. The same boilerplate sentence lands on both sides.
    tr, te = train_test_split(lab, test_size=0.2, random_state=SEED, stratify=lab["cat"])
    m = _model().fit(tr["cause_text"], tr["cat"])
    naive = f1_score(te["cat"], m.predict(te["cause_text"]), average="macro")
    seen = te["template"].isin(set(tr["template"])).mean()

    # 2. Honest: one row per distinct sentence, and every company is held out in turn.
    u = lab.drop_duplicates("template")
    pred = pd.Series(index=u.index, dtype=object)
    for tr_i, te_i in GroupKFold(n_splits=5).split(u, u["cat"], u["company"]):
        mm = _model().fit(u.iloc[tr_i]["cause_text"], u.iloc[tr_i]["cat"])
        pred.iloc[te_i] = mm.predict(u.iloc[te_i]["cause_text"])
    honest = f1_score(u["cat"], pred, average="macro")
    per_class = f1_score(u["cat"], pred, average=None, labels=sorted(u["cat"].unique()))
    by_company = (u.assign(ok=(pred == u["cat"])).groupby("company")["ok"].agg(["mean", "size"])
                  .query("size >= 10").sort_values("mean"))

    # 3. Use the model trained on all clear sentences to label the unclear and mixed ones.
    full = _model().fit(u["cause_text"], u["cat"])
    rest = ev[(ev["cat"] == "Unclear") | ev["amb"]].drop_duplicates("template")
    proba = full.predict_proba(rest["cause_text"])
    conf = proba.max(axis=1)
    return {
        "labelled_events": int(len(lab)),
        "labelled_sentences": int(len(u)),
        "naive_random_split_macro_f1": float(naive),
        "naive_test_sentences_seen_in_training": float(seen),
        "held_out_company_macro_f1": float(honest),
        "held_out_company_f1_by_class": dict(zip(sorted(u["cat"].unique()), map(float, per_class))),
        "worst_companies": [{"company": c, "accuracy": float(r["mean"]), "sentences": int(r["size"])}
                            for c, r in by_company.head(5).iterrows()],
        "unclear_or_mixed_sentences": int(len(rest)),
        "confidently_labelled_share": float((conf >= 0.6).mean()),
    }


def similarity_search(ev: pd.DataFrame, k: int = 5, dims: int = 128) -> tuple[dict, list[dict]]:
    """Embed each distinct cause sentence (TF-IDF reduced with SVD), index it in FAISS, and
    check whether the nearest sentences written by other companies share its cause label."""
    u = ev[(ev["cat"] != "Unclear") & ~ev["amb"]].drop_duplicates("template").reset_index(drop=True)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(ev["cause_text"].drop_duplicates())
    X = tfidf.transform(u["cause_text"])
    emb = normalize(TruncatedSVD(n_components=dims, random_state=SEED).fit_transform(X)).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _, nn = index.search(emb, 60)
    hits, examples = [], []
    comp, cat = u["company"].to_numpy(), u["cat"].to_numpy()
    for i in range(len(u)):
        others = [j for j in nn[i] if j != i and comp[j] != comp[i]][:k]
        if len(others) == k:
            hits.append(np.mean(cat[others] == cat[i]))
        if comp[i] == "Zoox" and len(examples) < 6 and others:
            j = others[0]
            examples.append({"zoox": u.loc[i, "cause_text"], "nearest_other": u.loc[j, "cause_text"],
                             "other_company": comp[j], "same_label": bool(cat[j] == cat[i])})
    base = (u["cat"].value_counts(normalize=True) ** 2).sum()  # chance two random sentences share a label
    return {"sentences": int(len(u)), "k": k, "precision_at_k": float(np.mean(hits)),
            "chance": float(base), "dims": dims}, examples
