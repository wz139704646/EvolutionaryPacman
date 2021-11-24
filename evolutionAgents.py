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

from game import Directions
from game import Agent
from game import Actions
from searchAgents import SearchAgent, PositionSearchProblem, FoodSearchProblem
from fitness import PacmanFitnessHelper, PositionFitnessHelper
import util
import time
import search
import random
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#########################################################
# This portion is written for you, but will only work   #
#     after you fill in parts of EvolutionSearchAgent   #
#########################################################

def rouletteWheel(indivs, num, key=lambda x:x):
    """select individuals by roulette wheel
    :param indivs: the individuals to select (sorted)
    :param number: number of selected individuals
    :param key: the key attribute to use
    """
    if len(indivs) == 0:
        return []

    vals = [key(indiv) for indiv in indivs]
    # relaxed lower bound
    baseVal = min(vals) * 0.9
    vals = [v - baseVal for v in vals]
    accVals = list(itertools.accumulate(vals))
    totVal = accVals[-1]

    selected = []
    for n in range(num):
        # take a random number from [0, totVal]
        r = random.uniform(0, totVal)
        for i in range(len(accVals)):
            if accVals[i] >= r:
                selected.append(indivs[i])
                break

        if len(selected) != n+1:
            # fixup
            selected.append(indivs[-1])

    return selected


def tournament(indivs, num, compSize, replace=False, key=lambda x:x):
    """tournament selection
    :param indivs: the individuals to select
    :param num: the number of individuals to be selected
    :param compSize: the number of individuals to compare
        (join tournament) each time
    :param replace: sample with replacement or not
    :param key: the key attribute used to compare (the bigger the better)
    """
    selected = []
    for _ in range(num):
        candidates = np.random.choice(range(len(indivs)), size=compSize, replace=replace)
        selected.append(indivs[candidates[np.argmax([key(indivs[c]) for c in candidates])]])

    return selected


class EvolutionSearchAgent():
    bestFits = []
    callFitnessTimes = []

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
        self.fitnessHelper = None
        self.best = None

    def parseOptions(self, options):
        """parse possible string options"""
        # seed for randomness
        self.seed = int(options['seed']) if 'seed' in options else None
        # dim of individuals
        self.actionDim = int(options['actionDim']) if 'actionDim' in options else 10
        # iterations for evolution
        self.T = int(options['T']) if 'T' in options else 200
        # number of individuals in the population
        self.popSize = int(options['popSize']) if 'popSize' in options else 100
        # size of mating pool (parents to select)
        self.poolSize = int(options['poolSize']) if 'poolSize' in options else 100
        # probability of crossover
        self.probCross = float(options['probCross']) if 'probCross' in options else 0.5
        # probability of mutation
        self.probMutation = float(options['probMutation']) if 'probMutation' in options else 0.05
        # the factor for fitness calculation
        self.fscale = float(options['fscale']) if 'fscale' in options else 0.1
        # the weight for some penalty term when calculating fitness
        self.penaltyWeight = float(options['penaltyWeight']) if 'penaltyWeight' in options else 0.0
        # the weight for some future related term when calculating fitness
        self.futureWeight = float(options['futureWeight']) if 'futureWeight' in options else 0.0
        # the log file (plot image) name
        self.logfile = options['logfile'] if 'logfile' in options else 'imgs/evolution.png'

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
            fitness = self.fitnessHelper.calculate(
                indiv, fscale=self.fscale,
                penaltyWeight=self.penaltyWeight,
                futureWeight=self.futureWeight)
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
        """record the best individual during the evolution"""
        curBest = self.population[
            np.argmax([indiv[1] for indiv in self.population])]

        if self.best is None or curBest[1] > self.best[1]:
            self.best = curBest

    def selectParents(self):
        """select parents into mating pool"""
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
            if util.flipCoin(self.probCross):
                offspring1, offspring2 = self.crossover(parent1[0], parent2[0])
            else:
                offspring1, offspring2 = parent1[0], parent2[0]

            # mutate
            if util.flipCoin(self.probMutation):
                offspring1 = self.mutate(offspring1)
            if util.flipCoin(self.probMutation):
                offspring2 = self.mutate(offspring2)

            self.offsprings += [offspring1, offspring2]

            i += 2

        if i < len(indices):
            indiv = self.matingPool[indices[i]][0]

            # mutate
            if util.flipCoin(self.probCross):
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
        """ensure the individual (maybe partial) is valid (the action is valid)"""
        end = end or len(indiv)
        for i in range(start, end):
            # get legal actions
            successors = self.problem.getSuccessors(indiv[i-1][0]) if i > 0 \
                else self.problem.getSuccessors(self.problem.getStartState())
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
        self.ensureValid(indiv, start=idx)

        return indiv

    def crossover(self, parent1, parent2):
        """crossover between two parents (without fitness)"""
        assert len(parent1) == len(parent2), "crossover require the same length genes"

        if len(parent1) <= 1:
            return parent1, parent2

        # one point crossover
        crossPoint = random.randint(1, len(parent1)-1)
        offs1 = parent1[:crossPoint] + parent2[crossPoint:]
        offs2 = parent2[:crossPoint] + parent1[crossPoint:]

        self.ensureValid(offs1, start=crossPoint)
        self.ensureValid(offs2, start=crossPoint)

        return offs1, offs2

    def stopCriterion(self):
        """evolution stop criterion"""
        if self.generationNum >= self.T:
            return True

        # can add other criterion here

        return False

    def initEvolution(self, problem):
        """initialize components for a new evolution"""
        self.problem = problem
        self.population = None
        self.matingPool = None
        self.offsprings = None
        self.best = None

        if isinstance(problem, PositionSearchProblem):
            self.fitnessHelper = PositionFitnessHelper(problem)
        else:
            self.fitnessHelper = PacmanFitnessHelper(problem, self.actionDim)

    def evolve(self, problem):
        """evolve solution for the problem"""
        self.initEvolution(problem)
        self.initPopulation()
        self.recordBest()

        print("========== Evolution begins ==========")

        EvolutionSearchAgent.bestFits.append(self.best[1])
        EvolutionSearchAgent.callFitnessTimes.append(self.numCallFitness)
        print('Generation Number: {}'.format(self.generationNum))
        print('Best fitness: {}'.format(self.best[1]))

        while not self.stopCriterion():
            # each generation
            self.selectParents()
            self.mate()
            self.survive()
            self.recordBest()

            EvolutionSearchAgent.bestFits.append(self.best[1])
            EvolutionSearchAgent.callFitnessTimes.append(self.numCallFitness)
            print('Generation Number: {}'.format(self.generationNum))
            print('Best fitness: {}'.format(self.best[1]))

        print("getFitness called times: {}".format(self.numCallFitness))
        print("========== Evolution ends ==========")

        # plot logs
        self.plot()

    def plot(self):
        """plot some log data"""
        plt.cla()
        plt.figure()
        fig, ax = plt.subplots()
        ax.plot(EvolutionSearchAgent.bestFits, color='green', label='fitness')
        ax.set_xlabel('number of evolutions')
        ax.set_ylabel('best fitness')
        ax1 = ax.twinx()
        ax1.plot(EvolutionSearchAgent.callFitnessTimes, linestyle='--', color='blue', label='call times')
        fitnessPatch = mpatches.Patch(lw=1, linestyle='-', color='green', label='fitness')
        timesPatch = mpatches.Patch(lw=1, linestyle='--', color='blue', label='call times')
        ax.legend(handles=[fitnessPatch, timesPatch])
        fig.savefig(self.logfile)
        plt.close(fig)

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


class EnhancedEvolutionSearchAgent(EvolutionSearchAgent):
    """the enhanced version of evolution search agent.
    Enhanced components:
    - survival selection: keep the elites or mating pool to the next population
    - parent selection: add tournament and rank methods
    """

    def parseOptions(self, options):
        super().parseOptions(options)

        # the number of elites in previous generation to keep
        self.elitesNum = int(options['elitesNum']) if 'elitesNum' in options else 0
        # whether to keep the mating pool to the next generation
        self.keepMatingPool = bool(options['keepMatingPool']) if 'keepMatingPool' in options else False
        # the type of selection
        self.selectionType = options['selectionType'] if 'selectionType' in options else 'roulette'
        # the size of tournament
        self.tournamentSize = int(options['tournamentSize']) if 'tournamentSize' in options else 20
        # sample with replacement or not when selecting parents
        self.selectReplace = bool(options['selectReplace']) if 'selectReplace' in options else True

    def selectParents(self):
        """select parents into mating pool"""
        if self.selectionType == 'roulette':
            # roulette wheel
            sortedPop = sorted(self.population, key=lambda x:x[1])
            self.matingPool = rouletteWheel(sortedPop, self.poolSize, lambda x:x[1])
        elif self.selectionType == 'tournament':
            # tournament selection
            self.matingPool = tournament(
                self.population, self.poolSize, self.tournamentSize,
                replace=self.selectReplace, key=lambda x:x[1])
        else:
            # rank
            sortedPop = sorted(self.population, key=lambda x:x[1], reverse=True)
            self.matingPool = sortedPop[:self.poolSize]

    def survive(self):
        # find the elites
        if self.elitesNum > 0:
            elites = sorted(self.population, key=lambda x:x[1], reverse=True)[:self.elitesNum]
        else:
            elites = []

        if self.keepMatingPool and self.popSize > self.poolSize:
            # keep the individuals in the mating pool to the next generation
            self.generationNum += 1
            sortedOffs = sorted(self.offsprings, key=lambda x:x[1], reverse=True)
            # keep both elites and mating pool
            self.population = elites + self.matingPool + sortedOffs
            self.population = self.population[:self.popSize]
        else:
            # keep the elites
            super().survive()
            self.population = elites + self.population[:self.popSize-self.elitesNum]

