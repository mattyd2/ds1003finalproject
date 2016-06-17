import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, datasets, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def analyzetext(df):
    alphas = np.logspace(-4, -.5, 30)
    X = df.judgedescript
    y = df.appl_decion_value

    tfidf_vct = TfidfVectorizer(stop_words='english', strip_accents='ascii',
                                ngram_range=(1, 3), decode_error='replace',
                                max_df=300, min_df=5, use_idf=True)
    X_train_tfidf = tfidf_vct.fit_transform(X)

    count_vct = CountVectorizer(binary=False, stop_words='english',
                                ngram_range=(1, 3), strip_accents='ascii',
                                decode_error='replace', max_df=300, min_df=5)
    X_train_count = count_vct.fit_transform(X)

    print "kfold using tfidf vectorization"
    lasso_cv = linear_model.LassoCV(alphas=alphas)
    k_fold = cross_validation.KFold(len(X), 10)
    for k, (train, test) in enumerate(k_fold):
        lasso_cv.fit(X_train_tfidf[train], y[train])
        print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
              format(k, lasso_cv.alpha_, lasso_cv.score(X_train_tfidf[test], y[test])))
    print "kfold using count vectorization"
    lasso_cv2 = linear_model.LassoCV(alphas=alphas)
    k_fold2 = cross_validation.KFold(len(X), 30)
    for k, (train, test) in enumerate(k_fold2):
        lasso_cv2.fit(X_train_tfidf[train], y[train])
        print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
              format(k, lasso_cv2.alpha_, lasso_cv2.score(X_train_count[test], y[test])))
