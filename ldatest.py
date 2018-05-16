import numpy as np
import math as math
import scipy as scipy
import copy
from scipy.special import gammaln, digamma
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import tarfile, os, argparse, getopt, sys

# rho = lambda: pow(1.0+1.0 , -0.5)


def dirichlet_expectation(ru): # array
    if (len(ru.shape) == 1):
        result = digamma(ru) - digamma(np.sum(ru))
    else:
        result = digamma(ru) - digamma(np.sum(ru, 1))[:, np.newaxis]
    return result.astype(ru.dtype)

def update_alpha(gamma_for_alpha, alpha, updateTime):
    doc_num = float(len(gamma_for_alpha))
    logphat = sum(dirichlet_expectation(ru) for ru in gamma_for_alpha) / doc_num
    # alpha_new = alpha.copy()
    gradient = doc_num * (digamma(np.sum(alpha)) - digamma(alpha) + logphat)
    c = doc_num * scipy.special.polygamma(1, np.sum(alpha))
    q = -doc_num * scipy.special.polygamma(1, alpha)
    b = np.sum(gradient / q) / ( 1. / c + np.sum(1 / q))
    alpha_new = -(gradient - b) / q
    rho = pow(1.0 + updateTime , -0.5) # decay ration = -0.5 by default setting
    if all(rho * alpha_new + alpha > 0):
        alpha += rho * alpha_new
    alpha = alpha/(np.sum(alpha)) # normalization
    return alpha

def update_ru(phi, N_d, eta, data, ru):
    for i in range(K):
        for j in range(V):
            sumPhiW = 0
            for d in range(M):
                for n in range(N_d[d]):
                    if data.indices[data.indptr[d]+n] == j:
                        sumPhiW += phi[d][n][i]*data.data[data.indptr[d]+n]
                    else:
                        sumPhiW += 0
            ru[i][j] = sumPhiW + eta
    return ru

def Estep (M, N_d, K, ru, alpha, data):
    tolerance = 0.001
    max_iter = 100
    gamma = np.random.gamma(100., 1. / 100., (M, K))
    phi = []
    for x in range(M):
        row = []
        for y in range(N_d[x]):
            col = [1. / M / N_d[x] / K for z in range(K)]
            col = np.asarray(col)
            row.append(col)
        row = np.asarray(row)
        phi.append(row)
    phi = np.asarray(phi)

    phi_new = phi.copy()
    gamma_new = gamma.copy()
    ExpElogBeta = np.exp(dirichlet_expectation(ru))

    for d in range(M):
        indices_beta = data.indices[data.indptr[d]:data.indptr[d + 1]]
        for a in range(max_iter):
            phi[d] = np.copy(phi_new[d])
            gamma[d] = np.copy(gamma_new[d])

            for n in range(N_d[d]):
                #print(n)
                for i in range(K):
                    phi_new[d][n][i] = ExpElogBeta[i][indices_beta[n]]*np.exp(scipy.special.digamma(gamma[d][i]))
                phi_new[d][n] = phi_new[d][n]/(np.sum(phi_new[d][n])) #normalization
                gamma_new[d] = alpha + phi_new[d][n] #update gamma_new
                # gamma_new[d] = gamma_new[d]/(np.sum(gamma_new[d])) # normalization

            if max(abs(gamma_new[d] - gamma[d])) < tolerance:
                phi[d] = np.copy(phi_new[d])
                gamma[d] = np.copy(gamma_new[d])
                # print(a)
                break
    return phi , gamma

def Mstep (phi, N_d, eta, data, gamma, alpha,ru, updateTime):
    ru = update_ru(phi, N_d, eta, data, ru)
    alpha = update_alpha(gamma, alpha, updateTime)
    return ru, alpha

def calc_perplexity(ru, gamma, N_d, data, alpha, eta, M, V):
    perplexity = 0.0
    ElogBeta = dirichlet_expectation(ru)
    ElogTheta = dirichlet_expectation(gamma)
    for d in range(M):
        sumSumLogExp = 0
        for n in range(N_d[d]):
            sumSumLogExp += data.data[data.indptr[d]+n] * np.log(np.sum(np.exp( ElogTheta[d, :] + ElogBeta[:, data.indices[data.indptr[d]+n]])))
        perplexity += sumSumLogExp
        perplexity += np.sum((alpha - gamma[d,:]) * ElogTheta[d,:])
        perplexity += np.sum(gammaln(gamma[d,:]) - gammaln(alpha))
        perplexity += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma[d,:]))
    perplexity += np.sum((eta - ru) * ElogBeta)
    perplexity += np.sum(gammaln(ru) - gammaln(eta))
    perplexity += np.sum( gammaln(eta * V) - gammaln(np.sum(ru, 1)) )
    perplexity = np.exp(-perplexity / data.sum())
    return perplexity

def dataRead(M, V): # read data
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data[: M]
    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=V,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    return tfidf

def combine_corpora(folder):
    chunks = []
    for file in os.listdir(folder):
        if file[0]!='.':
            lines = [line.strip() for line in open(folder+file)]
            chunk = ' '.join(lines)
            chunks.append(chunk)
    return chunks

# main
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "",
      ['M=', 'M-test='])

    for opt, arg in opts:
      if '--M'==opt:
        M = int(arg)
      elif '--M-test'==opt:
        M_test = int(arg)

    EPS = np.finfo(np.float).eps #prevent 0
    # M = 275 # documents in reuters (train1000)
    # M_test = 825 # documents in reuters (test25)
    # # M + M_test = 1025

    V = 3000 # words
    N = 30 # pick maximum N words
    K = 5 # topics

    #data = dataRead(M,V)

    chunks = combine_corpora(folder='./train_1800_200/')

    tfidf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                       max_features=V,
                                       stop_words='english')
    data_origin = tfidf_vectorizer.fit_transform(chunks)


    data = data_origin[0:M]
    data_test = data_origin[M:M+M_test]

    V = max(data.indices+1) # update the numbers of key words in vocabulary

    print(V)

    #initializations
    beta = np.asarray([[1./K/V for x in range(V)] for y in range(K)])
    alpha = np.asarray([1./K for x in range(K)])
    N_d = np.asarray([0 for x in range (M)])
    for a in range(M):
        N_d[a] = data.indptr[a+1]-data.indptr[a]
    N_d_test = np.asarray([0 for x in range (M_test)])
    for a in range(M_test):
        N_d_test[a] = data_test.indptr[a+1]-data_test.indptr[a]

    #for lambda
    eta = 1./K
    sstats = np.random.gamma(100., 1. / 100., (K, V))
    ru = eta + sstats

    #print(N_d_test)

    print(ru)
    print(alpha)
    # EM steps
    perplexity_data = []
    for updateTime in range(1,10,1):
        phi, gamma = Estep(M, N_d, K, ru, alpha, data)
        ru, alpha = Mstep(phi, N_d, eta, data, gamma, alpha, ru, updateTime)
        ### calculate perplexity
        phi, gamma_test = Estep(M_test, N_d_test, K, ru, alpha, data_test)
        perplexity = calc_perplexity(ru, gamma_test, N_d_test, data_test, alpha, eta, M_test, V)
        print(ru)
        print(alpha)
        print(perplexity)
        perplexity_data.append(perplexity)

    np.savez('beta_5.npz', ru, perplexity_data)
