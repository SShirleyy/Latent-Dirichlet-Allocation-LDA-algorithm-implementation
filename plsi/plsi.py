# coding:utf-8
# probabilistic latent semantic analysis
# yiming fan

import numpy as np
import os, sys, logging, argparse, getopt


class Corpus():
  def __init__(self, filename, dictionary, movie=False):
    self.filename = filename
    self.dictionary = dictionary # V
    if movie:
      self.movie = True
      self.corpora = np.load('./ml-latest-small/cf_matrix.npy')
    else:
      self.movie = False
      self.lines = [line.strip() for line in open(self.filename)] # D

      # TODO: add movielens-supported corpora format.
      self.corpora = np.zeros((len(self.lines), len(self.dictionary))) # (D, V)
      for index, line in enumerate(self.lines):
        words = line.split()
        for word in words:
          self.corpora[index][self.dictionary[word]]+=1
    logger.debug(self.corpora)

  def __iter__(self):
    if self.movie:
      for i in range(self.corpora.shape[0]):
        yield np.nonzero(self.corpora[i])[0].tolist()
    else:
      for line in self.lines:
        yield [self.dictionary[word] for word in line.split()]

  def __len__(self):
    if self.movie:
      return self.corpora.shape[0]
    else:
      return len(self.dictionary)


class PLSI:
  def __init__(self, corpus, movie=False):
    self.corpus = corpus

  def load(self, modelfile, logger=None):
    if not logger:
      logger = logging.getLogger('plsi')

    with open(modelfile, 'r') as fd:
      for line in (_.strip() for _ in fd):
        if line == "":
          break
        else:
          D, V, Z = [int(num) for num in line.split()]

      logger.info("load model: D {}, V {}, Z {}".format(D, V, Z))
      # define instance variables
      self.D = D
      self.V = V
      self.Z = Z # topics
      p_z = np.zeros(Z)
      p_v_z = np.zeros((V, Z))
      p_d_z = np.zeros((D, Z))

      for array in [p_z, p_v_z, p_d_z]:
        for line in (_.strip() for _ in fd):
          if line == "":
            break
          else:
            indexes, prob = line.split("\t")
            xs = [int(i) for i in indexes.split()]
            if len(xs) == 1:
              z = xs[0]
              array[z] = float(prob)
            else:
              x, z = xs
              array[x][z] = float(prob)

    # define instance variables
    self.p_z = p_z
    self.p_v_z = p_v_z
    self.p_d_z = p_d_z


  def predict(self, d, v):
    probs = [self.p_z[z]*self.p_v_z[v][z]*self.p_d_z[d][z] for z in range(self.Z)] # p(d,v|z)
    return np.argmax(probs)


def make_dictionary(filename):
  _id = 0
  dic = dict()
  for index, line in enumerate(open(filename)):
    logger.debug('line: {}'.format(index))
    line_ = line.strip().split()
    for word in line_:
      if word not in dic:
        dic[word] = _id
        _id+=1
  return dic


def train(corpus, Z: int, loop_count: int, logger=None):
  D = corpus.corpora.shape[0] # number of documents
  V = corpus.corpora.shape[1]
  W = np.sum(corpus.corpora) # number of words
  non_zero = np.nonzero(corpus.corpora)

  logger.info("Documents: {}, Vocabulary size: {}, Topics: {}, Words: {}".format(D, V, Z, W))

  p_z_dv = np.zeros((Z, D, V))
  p_v_z = np.zeros((V, Z))
  p_d_z = np.zeros((D, Z))
  p_z = np.zeros(Z)

  # initialize `p_z_dv` by dirichlet dist.
  for i in range(D):
    logger.debug('dirichlet row ({})'.format(i))
    for j in range(V):
      p_z_dv[:, i, j] = np.random.dirichlet(np.ones(Z), 1)[0]

  # train with EM.
  for i in range(loop_count):
    logger.debug('loop {}: M step'.format(i))
    for z in range(Z):
      denom = np.sum(p_z_dv[z]*corpus.corpora)
      p_z[z] = denom/W
      p_v_z[:, z] = [np.sum(p_z_dv[z, :, v]*corpus.corpora[:,  v])/denom for v in range(V)]
      p_d_z[:, z] = [np.sum(p_z_dv[z, d, :]*corpus.corpora[d, :])/denom for d in range(D)]

    logger.debug('loop {}: E step'.format(i))
    for d in range(D):
      for v in range(V):
        denom = sum(p_z[z] * p_d_z[d][z] * p_v_z[v][z] for z in range(Z))
        for z in range(Z):
          p_z_dv[z][d][v] = p_z[z] * p_d_z[d][z] * p_v_z[v][z]/denom

  logger.debug("sum of P(z): {}".format(p_z.sum()))
  logger.debug("sum of P(v|z)\n{}".format(p_v_z.T.sum(1)))
  logger.debug("sum of P(d|z)\n{}".format(p_d_z.T.sum(1)))

  # output
  print("{} {}\t{}".format(D, V, Z))
  print("")
  for z in range(Z):
    print("{}\t{:.6f}".format(z, p_z[z]))
  print("")
  for z in range(Z):
    for w in range(V):
      print("{} {}\t{:.6f}".format(w, z, p_v_z[w][z]))
  print("")
  for z in range(Z):
    for d in range(D):
      print("{} {}\t{:.6f}".format(d, z, p_d_z[d][z]))


if __name__ == "__main__":
  opts, args = getopt.getopt(sys.argv[1:], "",
    ['train', 'test', 'verbose', 'movie', 'file-path=', 'model-path=', 'num-topics='])

  if_train, if_test, if_verbose, if_movie = [False]*4
  file_path, model_path, num_topics = [None]*3

  for opt, arg in opts:
    if '--train'==opt:
      if_train = True
    elif '--test'==opt:
      if_test = True
    elif '--verbose'==opt:
      if_verbose = True
    elif '--movie'==opt:
      if_movie = True
    elif '--file-path'==opt:
      file_path = arg
    elif '--model-path'==opt:
      model_path = arg
    elif '--num-topics'==opt:
      num_topics = arg

  logger = logging.getLogger('plsi')
  logging.basicConfig(level=logging.DEBUG if if_verbose else logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  if if_train:
    print('well')
    if if_movie:
      corpus = Corpus(filename=None, dictionary=None, movie=True)
      train(corpus, Z=int(num_topics), loop_count=1, logger=logger)
    else:
      dictionary = make_dictionary(file_path)
      corpus = Corpus(file_path, dictionary)
      train(corpus, Z=int(num_topics), loop_count=10, logger=logger)

  if if_test:
    if if_movie:
      plsi = PLSI(corpus=None)
      plsi.load(model_path)

      perplexity = 0.0
      for d in range(plsi.D):
        perp_ = 0.0
        # foo = np.dot(plsi.p_d_z[d], )
        perp_ = np.prod(plsi.p_v_z[np.nonzero(plsi.p_v_z)])
        perp_ = perp_ if perp_!=0 else 1e-300
        perplexity+=np.log(perp_)
      logger.debug(perplexity)
      perplexity = np.exp(-perplexity/plsi.D)
      logger.debug('perplexity: {}'.format(perplexity))
      logger.debug(np.sum(plsi.p_v_z, axis=0))
    else:
      dictionary = make_dictionary(file_path)
      corpus = Corpus(file_path, dictionary)
      plsi = PLSI(corpus)#, dictionary)
      plsi.load(model_path)

      # calculate the perplexity.
      perplexity = 0.0
      for (d, line), words in zip(enumerate(corpus), (line.strip().split() for line in open(file_path))):
        perp_ = 0.0
        res, word_index_in_document = [[]]*2
        p_z_d = np.zeros(plsi.Z)

        for w, word in zip(line, words):
          word_index_in_document.append(w)
          i = plsi.predict(d, w) # make prediction.
          p_z_d[i]+=1
          # res.append("{}: {}, ".format(word, i))

        p_z_d = p_z_d/np.sum(p_z_d)
        foo = np.dot(plsi.p_v_z, p_z_d)
        perp_ = np.prod(foo[np.array(word_index_in_document)])
        perp_ = perp_ if perp_!=0 else 1e-200
        perplexity+=np.log(perp_)
        # print(''.join(res))
      logger.debug(perplexity)
      perplexity = np.exp(-perplexity/plsi.D)
      logger.debug('perplexity: {}'.format(perplexity))
      logger.debug(np.sum(plsi.p_v_z, axis=0))
