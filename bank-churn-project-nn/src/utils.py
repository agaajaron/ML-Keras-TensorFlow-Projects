# %%
from config import *


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="yellow")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()


def labeled_barplot(data, feature, perc=False, n=None):
    total = len(data[feature])
    count = data[feature].nunique()
    plt.figure(figsize=((n or count) + 1, 5))
    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data, x=feature, palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )
    for p in ax.patches:
        label = "{:.1f}%".format(100 * p.get_height() / total) if perc else p.get_height()
        ax.annotate(label, (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="center", size=12, xytext=(0, 5),
                    textcoords="offset points")
    plt.show()


def stratified_cv(X, y, clf_class, shuffle=True, **kwargs):
    stratified_k_fold = StratifiedKFold().split(X, y)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        Xtrain, Xtest = X.iloc[ii], X.iloc[jj]
        ytrain = y.iloc[ii]
        clf = clf_class(**kwargs)
        clf.fit(Xtrain, ytrain)
        y_pred.iloc[jj] = clf.predict(Xtest)
    return y_pred


def threshold_predict(model, X, threshold=0.5):
    proba = model.predict(X)
    return [1 if p > threshold else 0 for p in proba]
