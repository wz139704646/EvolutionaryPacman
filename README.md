# Evolutionary Pacman

Use evolution algorithm (GA) to play pacman.

## Codes

The main algorithm is in `evolutionAgents.py`.

The fitness calculation is in `fitness.py`.

## Usage

For example:
```shell
# the first version (use roulette wheel to select parents)
python pacman.py -l originalClassic -p EvolutionSearchAgent -a fscale=10,actionDim=5,popSize=20,poolSize=20,T=100,type=FoodSearchProblem,penalt yWeight=0.5,futureWeight=0.1,probMutation=0.3,probCross=0.7 -f -q -n 10

# the enhanced version (use tournament to select parents)
python pacman.py -l originalClassic -p EnhancedEvolutionSearchAgent -a seed=1126,fscale=10,actionDim=5,popSize=20,poolSize=20,T=100,type=FoodSearchProb lem,penaltyWeight=0.5,futureWeight=0.1,probMutation=0.3,probCross=0.7,selectionT ype=tournament,tournamentSize=2 -f -q -n 10
```

The arguments about EvolutionSearchAgent and EnhancedEvolutionSearchAgent (behind '-a') can be found at `EvolutionSearchAgent.parseOptions` and `EnhancedEvolutionSearchAgent.parseOptions`.

## Reulsts

Tested 10 times:
```shell
# the first version (use roulette wheel to select parents)
Average Score: 1428.3
Scores:        1615.0, 2826.0, 403.0, 1531.0, 2795.0, 1831.0, 2196.0, 45.0, 316.0, 725.0
Win Rate:      2/10 (0.20)
Record:        Loss, Win, Loss, Loss, Win, Loss, Loss, Loss, Loss, Loss

# the second version (use tournament to select parents)
Average Score: 1104.6
Scores:        3223.0, 977.0, 401.0, 686.0, 690.0, 1161.0, -329.0, -349.0, 1781.0, 2805.0
Win Rate:      2/10 (0.20)
Record:        Win, Loss, Loss, Loss, Loss, Loss, Loss, Loss, Loss, Win
```

