import util
import math
from game import Actions


def repeatedHeuristic(sequence):
    """
    Heurstic about goodness of a sequence by counting repeated number
    """
    return len(sequence) - len(set(sequence))


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


class FitnessHelper:
    """the class for fitness calculation"""

    def __init__(self) -> None:
        pass

    def calculate(self, indiv, **kwargs):
        raise NotImplementedError


class PositionFitnessHelper(FitnessHelper):
    """the class for fitness calculation in position search problem"""

    def __init__(self, problem):
        """initialize the help for fitness calculation of solution to a position search problem"""
        self.dists = {}
        self.problem = problem

        # initialize the true distances to the goal
        self.initDistances()

    def initDistances(self):
        """initialize the distance matrix to goal"""
        goal = self.problem.goal
        walls = self.problem.walls

        # the state to expand
        fringe = util.Queue()
        fringe.push((goal, 0))
        self.dists[goal] = 0

        while not fringe.isEmpty():
            pos, dist = fringe.pop()
            nbrs = Actions.getLegalNeighbors(pos, walls)

            for nbr in nbrs:
                if nbr in self.dists:
                    continue
                fringe.push((nbr, dist+1))
                self.dists[nbr] = dist + 1

    def calculate(self, indiv, **kwargs):
        """calculate fitness for a solution to position search problem"""
        totCost = 0 # the cost of the solution
        states = [] # the visited states

        for state, _, cost in indiv:
            # check each state
            totCost += cost
            states.append(state)
            if self.problem.isGoalState(state):
                break

        # distance to goal
        dist = self.dists[state] if state in self.dists else float('inf')

        if 'penaltyWeight' in kwargs:
            # wandering penalty
            wanderPenalty = kwargs['penaltyWeight'] + repeatedHeuristic(states)
            fitness = math.exp(-1 * kwargs['fscale'] * (totCost + dist + wanderPenalty))
        else:
            fitness = math.exp(-1 * kwargs['fscale'] * (totCost + dist))

        return fitness


class PacmanFitnessHelper(FitnessHelper):
    """the class for fitness calculation in complete pacman game"""

    def __init__(self, problem, actionDim):
        self.problem = problem
        self.actionDim = actionDim

        self.dangerZones = []
        self.initDangerZone()

    def initDangerZone(self):
        """initialize the danger zone of each ghost with variable range"""
        gameState = self.problem.startingGameState
        ghostStates = gameState.getGhostStates()

        for state in ghostStates:
            self.dangerZones.append(self.genDangerZone(state.getPosition(), self.actionDim))

    def genDangerZone(self, state, reach):
        """generate the danger zone around the state with the reach length"""
        walls = self.problem.walls

        dangerZone = {}
        fringe = util.Queue()
        fringe.push((state, 0))
        dangerZone[state] = 0

        for i in range(1, reach+1):
            # the edge of the zone
            curEdge = util.Queue()
            # the i-len danger zone
            while not fringe.isEmpty():
                pos, _ = fringe.pop()
                nbrs = Actions.getLegalNeighbors(pos, walls)

                for nbr in nbrs:
                    if nbr in dangerZone:
                        continue
                    curEdge.push((nbr, i))
                    dangerZone[nbr] = i

            while not curEdge.isEmpty():
                fringe.push(curEdge.pop())

        return dangerZone

    def calculate(self, indiv, **kwargs):
        # get game state
        gameState = self.problem.startingGameState
        walls = self.problem.walls
        _, initFoodGrid = self.problem.getStartState()
        initScore = gameState.getScore()
        dangerZones = self.dangerZones.copy()
        foodScore = 20
        scaredGhostScore = 200
        winScore = 500
        penaltyWeight = 1 if 'penaltyWeight' not in kwargs else kwargs['penaltyWeight']
        futureWeight = 1 if 'futureWeight' not in kwargs else kwargs['futureWeight']
        fscale = kwargs['fscale']

        # times of stepping into a danger zone
        dangerTimes = []

        curState = gameState
        for idx in range(len(indiv)):
            # check each inner state
            state, action, _ = indiv[idx]

            # generate successor game state and get useful information
            curState = curState.generatePacmanSuccessor(action)
            ghostStates = curState.getGhostStates()
            scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

            for i in range(len(scaredTimes)):
                gpos = ghostStates[i].getPosition()
                if gpos != gameState.getGhostStates()[i].getPosition():
                    # the state of the ghost has been changed
                    dangerZones[i] = self.genDangerZone(gpos, self.actionDim-i-1)

                if state[0] in dangerZones[i] and dangerZones[i][state[0]] <= i+1 and scaredTimes[i] == 0:
                    # step into a danger zone
                    dangerTimes.append(dangerZones[i][state[0]])

            if self.problem.isGoalState(state) or curState.isWin() or curState.isLose():
                # encounter goal state and terminate
                break

        # evaluate the final state
        position, foodGrid = state
        # game score
        score = curState.getScore()
        scoreChange = score - initScore
        # the distance to closet food
        minFoodDist = closestFood(position, foodGrid, walls)
        minFoodDist = minFoodDist if minFoodDist is not None else (walls.width * walls.height)
        # danger penalty sum
        dangerPenalty = sum([1/(d+1) for d in dangerTimes])

        # normalize
        minFoodDist /= walls.width * walls.height
        totScaredScore = scaredGhostScore * len(ghostStates)
        scoreChange /= (initFoodGrid.count() * foodScore + winScore + totScaredScore)

        fitness = math.exp(
            fscale *(scoreChange - penaltyWeight * dangerPenalty - futureWeight * minFoodDist))
        return fitness
