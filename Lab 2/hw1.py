from itertools import product
import math
import numpy as np
import random
from collections import defaultdict

from abc import ABCMeta, abstractmethod

# random.seed(42)


relevance = ["N", "R", "HR"]
rel_to_num = {"N":0, "R": 1, "HR":2}
r2binary = {"R": 1, "HR": 1, "N": 0}


### Simulate Rankings of Relevance for E and P

def create_rankings(k):
    
    possible_rankings = list(product(relevance, repeat=k*2))
    random.shuffle(possible_rankings)
    
    p = []
    e = []
    
    for ranking in possible_rankings:
        p.append(ranking[:k])
        e.append(ranking[k:])
        
    return p, e


p, e = create_rankings(5)
# print(p)
# print(e)

### Precision at rank k 

def precision(ordering, rank):
    tp = 0
    for i in range(rank):
        tp += r2binary[ordering[i]]
    return tp/rank


### Normalized Discounted Cumulative Gain at rank k (nDCG@k)

def DCG(ranking, rank):
    DCG = 0
    for i in range(rank):
        DCG += (math.pow(2, rel_to_num[ranking[i]]) - 1)/(math.log(1 + (i+1), 2))
    return DCG

### calculate normalizing constants for all ranks

normalizing_constants = []
for i in range(1, 6):
    normalizing_constants.append(DCG(['HR'] * i, i))# calculate DCG for best possible ranking (all HR)

def nDCG(ranking, rank):
    #divide DCG by normalizing constant appropriate to considerate rank
    return DCG(ranking, rank) / normalizing_constants[rank - 1]

### Expected Reciprocal Rank

def ERR(ranking, rank):
    p = 1
    ERR = 0
    
    def probability(r):
        tmp = np.power(2, r)
        return (tmp - 1) / np.max(tmp)
    
    ranking = np.array([rel_to_num[ranking[i]] for i in range(rank)])
    
    R = probability(ranking)
    
    for r in range(rank):
        ERR += (p * R[r]) / (r+1)
        p *= 1 - R[r]
    
    return ERR

#
# print(p[0])
# print("Precision: ", precision(p[0], 5))
# print("nDCG: ", nDCG(p[0], 5))
# print("ERR: ", ERR(p[0], 5))


# compare the rankings from Production and the ranking from Experimental with a given measure.

def compare(p, e, k, measure):
    
    assert len(p) == len(e)
    assert k > 0
    
    results = []
    
    for r in range(len(p)):
        m_e = measure(e[r], k)
        m_p = measure(p[r], k)

        results.append(m_e - m_p)
    
    return results

def compute(k, rankings=None):
    if rankings is None:
        p, e = create_rankings(k)
    else:
        p, e = rankings

    assert len(p) == len(e)
    assert k > 0

    measures = {'precision': precision, 'nDCG': nDCG, 'ERR': ERR}
    results = {}

    for name, measure in measures.items():
        res = compare(p, e, k, measure)
        results[name] = 0
        for result in res:
            if result >= 0:
                results[name] += 1
        # results[name] = [(i, r) for i, r in enumerate(res) if r > 0]
        results[name] /= len(res)

    return results

res = compute(5)
print(res, "\n")


# PART 2: Interleaving

### TEAM DRAFT INTERLEAVING
def teamDraftInterleaving(rankingA, rankingB):
    interleaved_list = []
    team_assignment = []

    ranking = [rankingA, rankingB]
    count = [0, 0]
    
    while count[0] < len(ranking[0]) and count[1] < len(ranking[1]):
        assignment = 1 - int(count[0] < count[1] or (count[0] == count[1] and np.random.rand() > 0.5))
        
        team_assignment.append(assignment)
        interleaved_list.append(ranking[assignment][count[assignment]])
        count[assignment] += 1
    
    return interleaved_list, team_assignment


### PROBABILISTIC INTERLEAVING
def probabilisticInterleaving(rankingA, rankingB):
    # COIN = 0 for rankingA,  COIN = 1 for rankingB
    k = len(rankingA)  # rank

    interleaved_list = []  # final list of merged documents
    team_assignment = []  # from which ordering does a document come
    used = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # mark already inserted documents
    
    pdoc = []  # probability distributions of documents
    tau = 3
    
    
    ### check if document was already inserted into the merged list
    def is_used(n, used):
        i = 0
        while n > pdoc[i]:
            i += 1
        return -1 if used[i] else i
    
    
    ### take a document from the ordering chose by the coin flip following the distribution probab. in that ordering
    def get_doc(used):
        rand = random.uniform(0, 1)
        idx = is_used(rand, used)
        while idx == -1:
            rand = random.uniform(0, 1)
            idx = is_used(rand, used)
        return idx
    
    
    ### create distributions of documents based on ranking
    for i in range(k):
        pdoc.append(pdoc[-1] + 1 / np.power(i + 1, tau) if len(pdoc) > 0 else 1 / np.power(i + 1, tau))
    pdoc /= pdoc[-1]

    
    ### toss coins and add a document to the new list
    for i in range(k * 2):
        
        if used[0].count(1) == k:
            coin = 1
        elif used[1].count(1) == k:
            coin = 0
        else:
            coin = random.randint(0, 1)
        
        idx = get_doc(used[coin])
        if coin:
            interleaved_list.append(rankingB[idx])
        else:
            interleaved_list.append(rankingA[idx])
        
        used[coin][idx] = 1
        team_assignment.append(coin)
        
    return interleaved_list, team_assignment


def computeCredit(interleaved_list, team_assignment, clicks):
    assert len(interleaved_list) == len(clicks)
    assert len(interleaved_list) == len(team_assignment)

    credits = 0, 0

    for team, n_clicks in zip(team_assignment, clicks):
        credits[team] += n_clicks

    return credits


interleaved_list, team_assignment = probabilisticInterleaving(p[0], e[0])
# print(p[0])
# print(e[0])
# print(interleaved_list)
# print(team_assignment)





# PART 3: Click Models

# Abstract Click Model Class
class UserClickModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, trainingset):
        pass

    @abstractmethod
    def probabilities(self, ranking):
        pass

    def click(self, probabilities):
        return np.random.binomial(n=1, p=probabilities)
    
    @abstractmethod
    def parse(self, filename):
        pass


# Random Click Model
class RandomClickModel(UserClickModel):

    def __init__(self):
        self.p = 0
    
    def train(self, trainingset):
        click_count, total_count = self.parse(trainingset)
        self.p = click_count/total_count
    
    def probabilities(self, ranking):
        return self.p * np.ones(len(ranking))

    def parse(self, filename):
        """
        Parses search sessions, formatted according to the Yandex Relevance Prediction Challenge (RPC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.
        An RPC file contains lines of two formats:
        1. Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action
        SessionID TimePassed TypeOfAction URLID
        
        :param filename: The name of the file with search sessions formatted according to RPC.
        :returns: total count
        """
    
        file = open(filename, "r")
        
        click_count = 0
        total_count = 0
        
        for line in file:
        
            entry_array = line.strip().split("\t")
        
            # If the entry has 6 or more elements it is a query
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                task = entry_array[0]
                docs = set(entry_array[5:])
                total_count += len(docs)
                
            # If the entry has 4 elements it is a click
            elif len(entry_array) == 4 and entry_array[2] == "C":
                doc = entry_array[3]
            
                if entry_array[0] == task and doc in docs:
                    click_count += 1
        
            # Else it is an unknown data format so leave it out
            else:
                continue
    
        return click_count, total_count


# Position-Based Model
class PositionBasedModel(UserClickModel):

    def __init__(self):
        self.gammas = []
        self.alphadict = {"N": 0.05, "R": 0.5, "HR": 0.95}
    
    def train(self, trainingset):
        documents, ranking = self.parse(trainingset)


        alphas = {}
        gammas = [0.1] * 10

        for document_id, document in documents.items():
            for query in document.keys():
                alphas[(document_id, query)] = 0.1

        epochs = 10

        for epoch in range(epochs):
            for document_id, document in documents.items():
                for query, occurrencies in document.items():
                    sum = 0
                    counter = 0

                    for list1 in occurrencies:
                        rank = list1[0]
                        click = list1[1]
                        if click:
                            counter += 1
                        alpha = alphas[(document_id, query)]
                        gamma = gammas[rank]

                        sum += click + (1 - click)*(1 - gamma)*alpha/(1 - gamma*alpha)

                    alphas[(document_id, query)] = sum / len(occurrencies)
                    #print("counterd o2c", counter)
                # query2sum = defaultdict(lambda: 0)
                # query2occurrencies = defaultdict(lambda: 0)
                # for query, ranking, click in document:
                #     alpha = alphas[(document, query)]
                #     gamma = gammas[ranking]
                #     query2sum[query] += click + (1 - click)*(1 - gamma)*alpha/(1 - gamma*alpha)
                #     query2occurrencies[query] += 1
                # for query, sum in query2sum.items():
                #     alphas[(document, query)] = sum/query2occurrencies[query]

            for i, rank in enumerate(ranking):
                sum = 0
                for query, document_id, click in rank:
                    alpha = alphas[(document_id, query)]
                    gamma = gammas[i]
                   # print((1 - alpha)*gamma / (1 - gamma*alpha))
                    sum += click + (1 - click)*(1 - alpha)*gamma / (1 - gamma*alpha)
                gammas[i] = sum / len(rank)

            print(epoch, gammas)
            print(alphas["1627", "1974"])               #(list(alphas.keys()))[0]])

        self.gammas = gammas

    
    def probabilities(self, ranking):
        attractiveness = [self.alphadict[ranking[rank]] for rank in range(len(ranking))]
        probabilities = np.array(attractiveness) * np.array(self.gammas[:len(ranking)])
        #print(probabilities)
        return probabilities



    
    def parse(self, filename):
        """
        Parses search sessions, formatted according to the Yandex Relevance Prediction Challenge (RPC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.
        An RPC file contains lines of two formats:
        1. Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action
        SessionID TimePassed TypeOfAction URLID
        
        :param filename: The name of the file with search sessions formatted according to RPC.

        :returns: 2 dictionaries:\n
        \t\t 1- an inverted index associating to each document the list of the queries in which it was returned, its ranking is those queries and whether it has been clicked or not\n
        \t\t 2- a list of 10 elements containing for each rank value (0 to 9, i.e. the indexes in the list) a list containing for each query the following tuple (query_id, doc_id, clicked)\n
        """
        
        file = open(filename, "r")
        
        ranking = [[] for rep in range(10)]
        
        documents = {}

        docs = None
        query_id = None
        task = None
        
        for line in file:
            
            entry_array = line.strip().split("\t")
            
            # If the entry has 6 or more elements it is a query
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                task = entry_array[0]
                query_id = entry_array[3]
                docs = entry_array[5:]
                
                for i, doc in enumerate(docs):

                    
                    if doc not in documents:
                        documents[doc] = {}
                    if query_id not in documents[doc]:
                        documents[doc][query_id] = []
                    
                    documents[doc][query_id].append([i, 0])
                    
                    ranking[i].append([query_id, doc, 0])

                docs = set(docs)

            # If the entry has 4 elements it is a click
            elif len(entry_array) == 4 and entry_array[2] == "C":
                doc = entry_array[3]
                
                if entry_array[0] == task and doc in docs:
                    rnk = documents[doc][query_id][-1][0]
                    documents[doc][query_id][-1][1] = 1
                    ranking[rnk][-1][2] = 1
            
            # Else it is an unknown data format so leave it out
            else:
                continue
        
        return documents, ranking


def modelPerformances(model, p, e, interleaver):
    wins = 0
    for index in range(len(p)):
        interleaved_list, team_assignment = interleaver(p[index], e[index])
        probs = model.probabilities(interleaved_list)



        clicks = model.click(probs)

        scores = [0, 0]  # p = 0, e = 1
        for j in range(len(clicks)):
            scores[team_assignment[j]] += clicks[j]

        wins += 1 if scores[1] >= scores[0] else 0

    return wins / len(p)


RCM = RandomClickModel()
PBM = PositionBasedModel()
RCM.train('YandexRelPredChallenge.txt')
PBM.train('YandexRelPredChallenge.txt')

n_experiments = 5

RCM_rate = 0
PBM_rate = 0
RCM_rate_p = 0
PBM_rate_p = 0
for experiment in range(n_experiments):

    RCM_rate += modelPerformances(RCM, p, e, teamDraftInterleaving)
    PBM_rate += modelPerformances(PBM, p, e, teamDraftInterleaving)
    #RCM_rate_p += modelPerformances(RCM, p, e, probabilisticInterleaving)
    #PBM_rate_p += modelPerformances(PBM, p, e, probabilisticInterleaving)

RCM_rate /= n_experiments
PBM_rate /= n_experiments
RCM_rate_p /= n_experiments
PBM_rate_p /= n_experiments

print("RCM: ", RCM_rate)
print("PBM: ", PBM_rate)
print("RCM_p: ", RCM_rate_p)
print("PBM_p: ", PBM_rate_p)


    
    