"""
Map-reduce algorithm to create a schedule of BPR samples.

The probability of emitting a candidate positive item
in the first mapper is designed to give a uniform
probability of any item in the dataset being output
as the positive item in the final list of triples.
"""

import random

J_IS_POSITIVE = '-'

class Mapper1:

    def __init__(self,user_item_counts,oversampling=1):
        self.N = sum(user_item_counts.values())  # number of non-zeros
        self.user_item_counts = user_item_counts
        self.max_item_count = max(user_item_counts.values())
        self.oversampling = oversampling

    def sample_positive(self,user):
        alpha = float(self.N-self.max_item_count)/(self.N-self.user_item_counts[user])
        return random.uniform(0,1) < alpha

    def rand_idx(self):
        return random.randint(0,self.N*self.oversampling)

    def __call__(self,user,item):
        # send candidate items to random indices
        for _ in xrange(self.oversampling):
            if self.sample_positive(user):
                # propose a candidate positive item
                yield self.rand_idx(),(user,item,'+')
            # propose a candidate negative item
            yield self.rand_idx(),(user,item,'-')

def reducer1(idx,values):
    # sample a positive and negative item uniformly to make a candidate triple
    seen = {'+':[],'-':[]}
    for user,item,c in values:
        seen[c].append((user,item))
    if seen['+'] and seen['-']:
        # we've got at least one postive and one negative item, now pick one
        pos = random.choice(seen['+'])
        neg = random.choice(seen['-'])
        yield (pos[0],neg[1]),pos[1]  # candidate triple as (u,j),i

def mapper2(user,item):
    # map the data again with an indicator value
    # to help us spot negative items in candidate triples
    yield (int(user),int(item)),J_IS_POSITIVE

def reducer2(key,values):
    user,j = key
    values = list(values)
    # check the positive items
    ii = set(i for i in values if i != J_IS_POSITIVE)
    if len(ii) == len(values):
        # j really is a negative item for u
        for i in ii:
            yield user,(i,j)

if __name__ == '__main__':

    import toydoop

    # create some user-item data
    data = {
            1:[10,20,30,40,50,60,70,80,90],
            2:[10,30,110,120,130,140,150],
            3:[20,30,40,90,120,160,170,180,190]
           }
    user_item_counts = dict((k,len(v)) for k,v in data.iteritems())

    datafile = 'bdoopr.in'
    mapout1 = 'bdoopr.map1'
    mapout2 = 'bdoopr.map2'
    outfile = 'bdoopr.out'

    f = open(datafile,'w')
    for user,items in data.iteritems():
        for item in items:
            print >>f,toydoop.default_formatter(user,item)
    f.close()

    # run two stages of mapreduce
    mapper1 = Mapper1(user_item_counts,oversampling=10)
    toydoop.mapreduce(datafile,mapout1,mapper=mapper1,reducer=reducer1)
    toydoop.mapreduce(datafile,mapout2,mapper=mapper2)  # map the data again
    toydoop.mapreduce([mapout1,mapout2],outfile,reducer=reducer2)
