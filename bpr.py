"""
Bayesian Personalized Ranking
"""

import numpy as np
from math import exp
import random

class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True,
                 sample_negative_items_empirically=False):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors
        self.sample_negative_items_empirically = sample_negative_items_empirically

class BPR(object):

    """
    D = number of factors
    """
    def __init__(self,D,args):
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        self.sample_negative_items_empirically = args.sample_negative_items_empirically

    """
    data = user-item matrix as a scipy sparse matrix
           useors and items are zero-indexed

           actually: it's just a list of lists of ids, no counts
           -- let's assume we use ints 1/0
    """
    def train(self,data,sampling_strategy,sample_negative_items_empirically,num_iters):
        self.init(data)

        print self.loss()
        for it in xrange(num_iters):
            print self.user_factors[:3,:]
            print self.item_factors[:3,:]
            print 'starting iteration {0}'.format(it)
            sampler = sampling_strategy(self.data,sample_negative_items_empirically)
            self.iterate(sampler)
            print self.loss()

    def iterate(self,sampler):
        for _ in xrange(self.data.nnz):
            u,i,j = sampler.sample_triple()
            self.update_factors(u,i,j)


    def init(self,data):
        self.data = data
        self.num_users,self.num_items = self.data.shape

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))

        # apply rule of thumb to decide num samples over which to compute loss
        num_triples = int(100*self.num_users**0.5)

        print 'sampling {0} <user,item i,item j> triples...'.format(num_triples)
        sampler = UniformUserUniformItem(data,self.sample_negative_items_empirically)
        self.loss_samples = [sampler.sample_triple() for _ in xrange(num_triples)]

    """
    apply SGD update
    """
    def update_factors(self,u,i,j,update_u=True,update_i=True):
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])

        z = 1.0/(1.0+exp(x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += 1.0/(1.0+exp(x))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return ranking_loss + 0.5*complexity

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])


# sampling strategies

class SamplingStrategy(object):

    def __init__(self,data,sample_negative_items_empirically):
        self.data = data
        self.sample_negative_items_empirically = sample_negative_items_empirically
        self.num_users,self.num_items = data.shape

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item(empirical=self.sample_negative_items_empirically)
        while j in user_items:
            j = self.random_item(empirical=self.sample_negative_items_empirically)
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    """
    sample an item uniformly or from the empirical distribution
    observed in the training data
    """
    def random_item(self,empirical=False):
        if empirical:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i

class UniformUserUniformItem(SamplingStrategy):

    def sample_triple(self):
        u = self.uniform_user()
        # sample positive item
        i = random.choice(self.data[u].indices)
        j = self.sample_negative_item(self.data[u].indices)
        return u,i,j

class UniformUserUniformItemWithoutReplacement(SamplingStrategy):

    def __init__(self,data,sample_negative_items_empirically):
        SamplingStrategy.__init__(self,sample_negative_items_empirically)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()

    def sample_triple(self):
        u = self.uniform_user()
        # sample positive item without replacement if we can
        user_items = self.local_data[u].nonzero()[1]
        if len(user_items) == 0:
            # reset user data if it's all been sampled
            for ix in self.local_data[u].indices:
                self.local_data[u,ix] = self.data[u,ix]
            user_items = self.local_data[u].nonzero()[1]
        i = random.choice(user_items)
        # forget this item so we don't sample it again for the same user
        self.local_data[u,i] = 0
        j = self.sample_negative_item(user_items)
        return u,i,j

class UniformPair(SamplingStrategy):

    def __init__(self,data,sample_negative_items_empirically):
        SamplingStrategy.__init__(self,data,sample_negative_items_empirically)
        self.users,self.items = self.data.nonzero()

    def sample_triple(self):
        idx = random.randint(0,self.data.nnz-1)
        u = self.users[self.idx]
        i = self.items[self.idx]
        j = self.sample_negative_item(self.data[u])
        return u,i,j

class UniformPairWithoutReplacement(SamplingStrategy):

    def __init__(self,data,sample_negative_items_empirically):
        SamplingStrategy.__init__(self,data,sample_negative_items_empirically)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users,self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0

    def sample_triple(self):
        u = self.users[self.idx]
        i = self.items[self.idx]
        j = self.sample_negative_item(self.data[u])
        self.idx += 1
        return u,i,j

if __name__ == '__main__':

    # learn a matrix factorization with BPR like this:

    import sys
    from scipy.io import mmread

    data = mmread(sys.argv[1]).tocsr()

    args = BPRArgs()
    args.learning_rate = 0.2

    num_factors = 10
    model = BPR(num_factors,args)

    sampling_strategy = UniformPairWithoutReplacement
    sample_negative_items_empirically = True
    num_iters = 10
    model.train(data,sampling_strategy,sample_negative_items_empirically,num_iters)
