"""
Precompute a schedule of samples and use it to train
a BPR model.

Sampling is done in a map-reduce style.
"""

from bdoopr import *
import toydoop
from bpr import BPR, BPRArgs, ExternalSchedule
from numpy import loadtxt
from scipy.sparse import coo_matrix
import sys

def parser(line):
    return map(int,line.strip().split()[:2])

def formatter(key,val):
    return '{0}\t{1}\t{2}'.format(key,val[0],val[1])

datafile = sys.argv[1]  # one-indexed, whitespace separated
sample_file = datafile+'.samples'
tmp1 = sample_file+'.tmp1'
tmp2 = sample_file+'.tmp2'

print 'reading data...'
data = loadtxt(datafile)
print 'converting to zero-indexed sparse matrix...'
idxs = data[:,:2]-1
vals = data[:,2]
data = coo_matrix((vals,idxs.T)).tocsr()
user_item_counts = dict((i+1,data[i].getnnz()) for i in xrange(data.shape[0]))

print 'creating samples...'
mapper1 = Mapper1(user_item_counts,oversampling=10)
print 'map-red1...'
toydoop.mapreduce(datafile,tmp1,mapper=mapper1,reducer=reducer1,parser=parser)
print 'map2...'
toydoop.mapreduce(datafile,tmp2,mapper=mapper2,parser=parser)  # map the data again
print 'red2...'
toydoop.mapreduce([tmp1,tmp2],sample_file,reducer=reducer2,formatter=formatter)

print 'training...'
args = BPRArgs()
args.learning_rate = 0.3
num_factors = 10
model = BPR(num_factors,args)
sampler = ExternalSchedule(sample_file,index_offset=1)  # schedule is one-indexed
num_iters = 10
model.train(data,sampler,num_iters)
