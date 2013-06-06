"""
Sparse LInear Method for collaborative filtering
using BPR to optimise for AUC.

Uses pysparse for the item similarity matrix as
this appears to give much faster read/write access
than anything in scipy.sparse.  Easiest to install
the Common Sense Computing version like this:
  pip install csc-pysparse
The data is still held in a scipy.sparse.csr_matrix
as for the access we need here that is actually faster
than the pysparse ll_mat.
"""

from pysparse.sparse.spmatrix import *
import numpy as np
from math import exp

from bpr import BPRArgs, ExternalSchedule

class BPRSLIM(object):

    def __init__(self,args):
        """
        initialise SLIM model
        """
        self.learning_rate = args.learning_rate
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.init_similarity_std = 0.1

    def train(self,data,sampler,num_iters):
        """train model
        data:    user-item matrix as a scipy sparse matrix
                 users and items are zero-indexed
        sampler: must be ExternalSchedule
        """
        self.data = data
        self.num_users,self.num_items = self.data.shape

        print 'finding sparsity structure of item similarities...'
        # do a dry run of an iteration and collect the item indices
        indices = set()
        for  u,i,j in sampler.generate_samples(self.data):
            for l in self.data[u].indices:
                if l != i:
                    indices.add((i,l))
                if l != j:
                    indices.add((j,l))
        print 'initialising item similarities...'
        self.item_similarities = ll_mat(self.num_items,self.num_items,len(indices))
        indices = np.array(list(indices))
        ii = indices[:,0]
        jj = indices[:,1]
        vals = self.init_similarity_std * np.random.randn(len(indices))
        for i,j,v in zip(ii,jj,vals):
            self.item_similarities[int(i),int(j)] = v

        # TODO: with pysparse we *might* get away with lazy initialization
        # and letting the item similarities grow over time...
        # i.e. we wouldn't be tied to a fixed schedule

        # create loss samples, again restrict to the scheduled samples
        # so we have initialised item similarities
        num_loss_samples = int(100*self.num_users**0.5)
        self.loss_samples = [t for t in sampler.generate_samples(data,num_loss_samples)]

        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            for u,i,j in sampler.generate_samples(self.data):
                self.update_factors(u,i,j)
            print 'iteration {0}: loss = {1}'.format(it,self.loss())

    def loss(self):

        # TODO: this seems to take a lot of the traning time - why??

        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += 1.0/(1.0+exp(x))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.positive_item_regularization * self.item_similarities[i,:].norm('fro')**2
            complexity += self.negative_item_regularization * self.item_similarities[j,:].norm('fro')**2

        return ranking_loss + 0.5*complexity

    def predict(self,u,i):
        return sum(self.item_similarities[i,int(l)] for l in self.data[u].indices if l != i)

    def update_factors(self,u,i,j):
        """apply SGD update"""

        x = sum(self.item_similarities[i,int(l)]-self.item_similarities[j,int(l)] for l in self.data[u].indices)

        z = 1.0/(1.0+exp(x))

        # update item similarity weights
        for l in self.data[u].indices:
            l = int(l)
            if l != i:
                d = z - self.positive_item_regularization*self.item_similarities[i,l]
                self.item_similarities[i,l] += self.learning_rate*d
            if l != j:
                d = -z - self.negative_item_regularization*self.item_similarities[j,l]
                self.item_similarities[j,l] += self.learning_rate*d

if __name__ == '__main__':

    # learn SLIM item similarities with BPR like this:

    import sys
    from scipy.io import mmread

    data = mmread(sys.argv[1]).tocsr()
    sample_file = sys.argv[2]

    args = BPRArgs()
    args.learning_rate = 0.3

    model = BPRSLIM(args)

    num_iters = 10
    sampler = ExternalSchedule(sample_file,index_offset=1)  # schedule is one-indexed

    model.train(data,sampler,num_iters)

