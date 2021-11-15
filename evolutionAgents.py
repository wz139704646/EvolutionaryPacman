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
import util
import time
import search
import random


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


class EvolutionSearchAgent():
    ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    def __init__(
        self, type='PositionSearchProblem', actionDim=10,
        T=100, popSize=50, probCross=0.5, probMutation=0.05):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        :param type: the problem type (class name) to solve
        :param actionDim: the numebr of actions generated per search (the whole evolution)
        :param T: the number of iterations for one evolution
        :param popSize: the size of the population
        '''
        self.searchType = globals()[type]
        self.actionDim = actionDim # dim of individuals
        self.T = T # iterations for evolution
        self.popSize = popSize # number of individuals in the population
        self.probCross = probCross
        self.probMutation = probMutation

        self.population = None
        self.numCallFitness = 0
        self.matingPool = None

    def getFitness(self, state, problem):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        self.numCallFitness += 1

        return positionHeuristic(state, problem)

    def initPopulation(self, problem):
        """initialize the population
        :param problem: the problem to evolve for
        """
        self.population = []
        self.generationNum = 0

        state = problem.getStartState()
        # random walk through the tree
        for _ in range(self.popSize):
            individual = []
            cur_state = state

            for _ in range(self.actionDim):
                successors = problem.getSuccessors(cur_state)
                s = random.choice(successors)
                individual.append(s)
                cur_state = s[0]

    def selectParents(self):
        pass

    def survive(self):
        pass

    def mate(self, parents):
        pass

    def mutation(self):
        pass

    def crossover(self):
        pass

    def stopCriterion(self):
        return self.generationNum >= self.T

    def evolve(self, problem):
        """evolve solution for the problem"""
        startState = problem.getStartState()

        self.initPopulation(problem)
        while not self.stopCriterion():
            # each generation
            parents = self.selectParents()
            self.mate(parents)
            self.survive()

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
        pass

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
                action = Directions.STOP

            return action
