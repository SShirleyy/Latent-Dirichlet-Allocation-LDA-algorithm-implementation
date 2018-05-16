# coding:utf-8
# mixture of unigrams

import numpy as np
import os, sys, logging, argparse, getopt


class Corpus():
  def __init__(self, filename, dictionary):
    self.filename = filename
    self.dictionary = dictionary # V
    self.lines = [line.strip() for line in open(self.filename)] # D
    self.D = len(self.lines)

    self.corpora = np.zeros(len(self.dictionary)) # V
    for index, line in enumerate(self.lines):
      words = line.split()
      for word in words:
        self.corpora[self.dictionary[word]]+=1
    logger.debug(self.corpora)

  def __iter__(self):
    for line in self.lines:
      yield [self.dictionary[word] for word in line.split()]

  def __len__(self):
    return len(self.dictionary)


class PLSI:
  def __init__(self, corpus):
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

      for array in [p_z, p_v_z]:
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


  def predict(self, d, v): # TODO
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
  V = corpus.corpora.shape[0]
  W = np.sum(corpus.corpora) # number of words
  non_zero = np.nonzero(corpus.corpora)

  logger.info("Vocabulary size: {}, Topics: {}, Words: {}".format(V, Z, W))

  p_z_v = np.zeros((Z, V)) # p(z|v)
  p_v_z = np.zeros((V, Z)) # p(v|z)
  p_z = np.zeros(Z) # p(z)

  # initialize `p_z_v` by dirichlet dist.
  for i in range(V):
    logger.debug('dirichlet row ({})'.format(i))
    p_z_v[:, i] = np.random.dirichlet(np.ones(Z), 1)[0]

  # train with EM.
  for i in range(loop_count):
    logger.debug('loop {}: M step'.format(i))
    for z in range(Z):
      denom = np.sum(p_z_v[z]*corpus.corpora)
      p_z[z] = denom/W
      p_v_z[:, z] = [p_z_v[z, v]*corpus.corpora[v]/denom for v in range(V)]

    logger.debug('loop {}: E step'.format(i))
    for v in range(V):
      evid = np.sum(p_z*p_v_z[v])
      # denom = sum(p_z[z]*p_v_z[v][z] for z in range(Z))
      for z in range(Z):
        p_z_v[z][v] = p_z[z]*p_v_z[v][z]/evid

  logger.debug("sum of P(z): {}".format(p_z.sum()))
  logger.debug("sum of P(v|z)\n{}".format(p_v_z.T.sum(1)))

  # output
  print("{} {}\t{}".format(corpus.D, V, Z))
  print("")
  for z in range(Z):
    print("{}\t{:.6f}".format(z, p_z[z]))
  print("")
  for z in range(Z):
    for w in range(V):
      print("{} {}\t{:.6f}".format(w, z, p_v_z[w][z]))


if __name__ == "__main__":
  opts, args = getopt.getopt(sys.argv[1:], "",
    ['train', 'test', 'verbose', 'file-path=', 'model-path=', 'num-topics='])

  if_train, if_test, if_verbose = [False]*3
  file_path, model_path, num_topics = [None]*3

  for opt, arg in opts:
    if '--train'==opt:
      if_train = True
    elif '--test'==opt:
      if_test = True
    elif '--verbose'==opt:
      if_verbose = True
    elif '--file-path'==opt:
      file_path = arg
    elif '--model-path'==opt:
      model_path = arg
    elif '--num-topics'==opt:
      num_topics = arg

  logger = logging.getLogger('plsi')
  logging.basicConfig(level=logging.DEBUG if if_verbose else logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  if if_train:
    dictionary = make_dictionary(file_path)
    corpus = Corpus(file_path, dictionary)
    train(corpus, Z=int(num_topics), loop_count=10, logger=logger)

  if if_test:
    dictionary = make_dictionary(file_path)
    corpus = Corpus(file_path, dictionary)
    plsi = PLSI(corpus)#, dictionary)
    plsi.load(model_path)

    # calculate the perplexity.
    perplexity = 0.0
    for (d, line), words in zip(enumerate(corpus), (line.strip().split() for line in open(file_path))):
      perp_ = 0.0
      res, word_index_in_document = [[]]*2

      for w, word in zip(line, words):
        word_index_in_document.append(w)
        # i = plsi.predict(d, w)
        # p_z_d[i]+=1
        # res.append("{}: {}, ".format(word, i))

      for z in range(plsi.Z):
        perp_+=np.prod(plsi.p_v_z[word_index_in_document, z])*plsi.p_z[z]
        logger.debug(perp_)
      perp_ = perp_ if perp_!=0 else 1e-320
      perplexity+=np.log(perp_)
      # print(''.join(res))
    logger.debug(perplexity)
    perplexity = np.exp(-perplexity/plsi.D)
    logger.debug('perplexity: {}'.format(perplexity))
    logger.debug(np.sum(plsi.p_v_z, axis=0))
