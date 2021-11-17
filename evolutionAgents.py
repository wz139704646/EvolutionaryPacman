# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
A* search , run the following command:

> python pacman.py -l smallmaze -p EvolutionSearchAgent

"""

from numpy.core.numeric import indices
from game import Directions
from game import Agent
from game import Actions
from searchAgents import SearchAgent, PositionSearchProblem, FoodSearchProblem
import util
import time
import search
import random
import math
import itertools
import numpy as np


#########################################################
# This portion is written for you, but will only work   #
#     after you fill in parts of EvolutionSearchAgent   #
#########################################################

def positionHeuristic(state, problem):
    '''
    A simple heuristic function for evaluate the fitness in task 1
    :param state:
    :return:
    '''
    return util.manhattanDistance(state, problem.goal)


def repeatedHeuristic(sequence):
    """
    Heurstic about goodness of a sequence by counting repeated number
    """
    nodeSet = set()
    newNum = 0

    for n in sequence:
        if n not in nodeSet:
            nodeSet.add(n)
            newNum += 1

    return len(sequence) - newNum


def rouletteWheel(indivs, num, key=lambda x:x):
    """select individuals by roulette wheel
    :param indivs: the individuals to select
    :param number: number of selected individuals
    :param key: the key attribute to use
    """
    vals = [key(indiv) for indiv in indivs]
    accVals = list(itertools.accumulate(vals))
    totVal = accVals[-1]

    selected = []
    for _ in range(num):
        # take a random number from [0, totVal]
        r = random.uniform(0, totVal)
        for i in range(len(accVals)):
            if accVals[i] >= r:
                selected.append(indivs[i])
                break

    return selected


class EvolutionSearchAgent():
    def __init__(self, type='PositionSearchProblem', **kwargs):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        :param type: the problem type (class name) to solve
        :param actionDim: the numebr of actions generated per search (the whole evolution)
        :param T: the number of iterations for one evolution
        :param popSize: the size of the population
        '''
        self.searchType = globals()[type]
        self.parseOptions(kwargs)

        # set seed
        if self.seed is not None:
            random.seed(self.seed)

        self.problem = None
        self.population = None
        self.numCallFitness = 0
        self.matingPool = None
        self.offsprings = None
        self.best = None

    def parseOptions(self, options):
        """parse possible string options"""
        # seed for randomness
        self.seed = int(options['seed']) if 'seed' in options else None
        # dim of individuals
        self.actionDim = int(options['actionDim']) if 'actionDim' in options else 10
        # iterations for evolution
        self.T = int(options['T']) if 'T' in options else 100
        # number of individuals in the population
        self.popSize = int(options['popSize']) if 'popSize' in options else 50
        # size of mating pool (parents to select)
        self.poolSize = int(options['poolSize']) if 'poolSize' in options else 50
        # probability of crossover
        self.probCross = float(options['probCross']) if 'probCross' in options else 0.5
        # probability of mutation
        self.probMutation = float(options['probMutation']) if 'probMutation' in options else 0.05
        # the factor for fitness calculation
        self.fscale = float(options['fscale']) if 'fscale' in options else 0.1

    def getFitness(self, individuals):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        self.numCallFitness += 1

        extendedIndivs = [] # store the individuals extended with the fitness
        for indiv in individuals:
            # count the cost to take
            totCost = 0
            for s in indiv:
                totCost += s[2]
                if self.problem.isGoalState(s[0]):
                    # encounter goal state and terminate
                    break

            # wandering penalty
            wanderingPenalty = repeatedHeuristic([s[0] for s in indiv])

            fitness = math.exp(
                -1 * self.fscale * (totCost + wanderingPenalty + positionHeuristic(
                    s[0], self.problem)))
            extendedIndivs.append((indiv, fitness))

        return extendedIndivs

    def initPopulation(self):
        """initialize the population
        :param problem: the problem to evolve for
        """
        self.population = []
        self.generationNum = 0

        state = self.problem.getStartState()
        # random walk through the tree
        for _ in range(self.popSize):
            individual = []
            curState = state

            for _ in range(self.actionDim):
                successors = self.problem.getSuccessors(curState)
                s = random.choice(successors)
                individual.append(s)
                curState = s[0]
            self.population.append(individual)

        self.population = self.getFitness(self.population)

    def recordBest(self):
        curBest = self.population[
            np.argmax([indiv[1] for indiv in self.population])]

        if self.best is None or curBest[1] > self.best[1]:
            self.best = curBest

    def selectParents(self):
        sortedPop = sorted(self.population, key=lambda x:x[1])
        self.matingPool = rouletteWheel(sortedPop, self.poolSize, lambda x:x[1])

    def mate(self):
        """mate parents to generate offsprings"""
        indices = list(range(self.poolSize))
        random.shuffle(indices)

        # generate offsprings
        self.offsprings = []
        i = 0
        while i+1 < len(indices):
            parent1 = self.matingPool[indices[i]]
            parent2 = self.matingPool[indices[i+1]]

            # crossover
            if random.random() < self.probCross:
                offspring1, offspring2 = self.crossover(parent1[0], parent2[0])
            else:
                offspring1, offspring2 = parent1[0], parent2[0]

            # mutate
            if random.random() < self.probMutation:
                offspring1 = self.mutate(offspring1)
            if random.random() < self.probMutation:
                offspring2 = self.mutate(offspring2)

            self.offsprings += [offspring1, offspring2]

            i += 2

        if i < len(indices):
            indiv = self.matingPool[indices[i]]

            # mutate
            if random.random() < self.probCross:
                indiv = self.mutate(indiv)

            self.offsprings.append(indiv)

        # calculate the fitness of offsprings
        self.offsprings = self.getFitness(self.offsprings)

    def survive(self):
        """survival competition between parents and offsprings"""
        self.generationNum += 1

        if self.popSize == self.poolSize:
            # age-based
            self.population = self.offsprings
        elif self.popSize > self.poolSize:
            # replace worst
            sortedPop = sorted(self.population, key=lambda x:x[1], reverse=True)
            self.population = self.offsprings + sortedPop[:self.popSize-self.poolSize]
        else:
            # (\mu, \lambda) selection
            sortedOffs = sorted(self.offsprings, key=lambda x:x[1], reverse=True)
            self.population = sortedOffs[:self.popSize]

    def ensureValid(self, indiv, start=0, end=None):
        """ensuer the individual (maybe partial) is valid (the action is valid)"""
        end = end or len(indiv)
        for i in range(start+1, end):
            # get legal actions
            successors = self.problem.getSuccessors(indiv[i-1][0])
            for s in successors:
                # if the same action found, replace the entire node
                if indiv[i][1] == s[1]:
                    indiv[i] = s
                    break

            # no nodes with the same action, randomly choose one
            if indiv[i] != s:
                indiv[i] = random.choice(successors)

    def mutate(self, indiv):
        """mutate a individual (without fitness)"""
        # random select a node on the path
        idx = random.choice(list(range(len(indiv))))

        # mutate the node by regenerate it randomly
        # the previous nodes
        nodes = [(self.problem.getStartState(),)] + indiv
        node = nodes[idx] # find the previous nodes
        # regenerate
        successors = self.problem.getSuccessors(node[0])
        node = random.choice(successors)

        indiv[idx] = node
        self.ensureValid(indiv)

        return indiv

    def crossover(self, parent1, parent2):
        """crossover between two parents (without fitness)"""
        # one point crossover
        crossPoint = random.randint(1, len(parent1)-1)
        offs1 = parent1[:crossPoint] + parent2[crossPoint:]
        offs2 = parent2[:crossPoint] + parent1[crossPoint:]

        self.ensureValid(offs1)
        self.ensureValid(offs2)

        return offs1, offs2

    def stopCriterion(self):
        if self.generationNum >= self.T:
            return True

        # check whether exist the optimal solution now
        if isinstance(self.problem, PositionSearchProblem):
            bestFitness = math.exp(
                -1 * self.fscale * positionHeuristic(
                    self.problem.getStartState(), self.problem))
            if self.best[1] >= bestFitness:
                return True

        return False

    def evolve(self, problem):
        """evolve solution for the problem"""
        self.problem = problem
        self.population = None
        self.matingPool = None
        self.offsprings = None
        self.best = None

        self.initPopulation()
        self.recordBest()
        print('Generation Number: {}'.format(self.generationNum))
        print('Best fitness: {}'.format(self.best[1]))

        while not self.stopCriterion():
            # each generation
            self.selectParents()
            self.mate()
            self.survive()
            self.recordBest()

            print('Generation Number: {}'.format(self.generationNum))
            print('Best fitness: {}'.format(self.best[1]))

    def generateLegalActions(self):
        '''
        generate the individuals with legal actions
        :return:
        '''
        pass

    def getActions(self, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param problem:
        :return: the best individual in the population
        '''
        self.evolve(problem)

        actions = [s[1] for s in self.best[0]]
        print("Solution: {}".format(actions))

        return actions

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.getActions(problem)  # Find a path

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions): # You may need to add some conditions for taking the action
            return self.actions[i]
        else: # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(problem)
            if len(self.actions) > 0:
                action = self.actions[self.actionIndex]
                self.actionIndex += 1
            else:
                print("STOP")
                action = Directions.STOP

            return action
