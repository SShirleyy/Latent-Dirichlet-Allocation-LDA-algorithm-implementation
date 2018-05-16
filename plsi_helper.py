# coding:utf-8

'''
  plsi_helper.py
'''


import os, sys


def main(argv):
  pass


def combine_corpora(folder):
  for file in os.listdir(folder):
    if file[0]!='.':
      lines = [line.strip() for line in open(folder+file)]
      chunk = ' '.join(lines)
      print(chunk)


if __name__ == '__main__':
  # main(sys.argv)
  combine_corpora(folder='./reuters/training/')