import numpy as np
from backend import *
import signal
import sys
import matplotlib

# Erzwinge das Tkinter-Backend
matplotlib.use("TkAgg")

def signal_handler(sig, frame):
    print("Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class SimpleUtility(UtilityFunction):
    def calculate(self, action, favor_size):
        """
        Simple utility calculation:
        - Cooperation: Requester gains +favor_size, responder loses -favor_size.
        - Rejection: No utility change.
        """
        if action == "cooperate":
            return favor_size, -favor_size/2
        elif action == "reject":
            return -favor_size/3, 0
        else:  # No action
            return 0, 0
        
class ReputationManager:
    def __init__(self, gain_base=0.1, loss_base=0.1, min_reputation=-1.0, max_reputation=1.0):
        self.gain_base = gain_base
        self.loss_base = loss_base
        self.min_reputation = min_reputation
        self.max_reputation = max_reputation
        self.reputation_scaler = 1.5

    def update_reputation(self, asking, helping, action, favor_size):
        if helping.get_average_utility_per_round() < 0: # Edited for promoting lonewolves. Comment out for old program
            reputation_change_asking = self.loss_base * favor_size * (1 + asking.real_reputation/self.reputation_scaler)/3
            asking.real_reputation = max(self.min_reputation, asking.real_reputation - reputation_change_asking)
        if action == "accept":
            reputation_change = self.gain_base * favor_size
            helping.real_reputation = min(self.max_reputation, helping.real_reputation + reputation_change)
        elif action == "reject":
            reputation_change = self.loss_base * favor_size * (1 + asking.real_reputation/self.reputation_scaler)
            helping.real_reputation = max(self.min_reputation, helping.real_reputation - reputation_change)
        helping.public_reputation = 1 if helping.real_reputation >= 0 else -1

if __name__ == "__main__":
    L = 10  # Grid size
    N = 1   # Neighborhood radius

    strategy_generator_instance = StrategyGenerator(
        favor_sizes=[1, 3],
        reputation_values=[-1, 1]
    )

    #print([(strategie.bitcode, strategie.moral_score) for strategie in strategy_generator_instance.generate_all_strategies()])

    grid = GameGrid(L, N, strategy_generator_instance, diagonal_neighbors=True)
    '''#own_grid = [
    "110000", "110000", "110000", "110000", "110000", "110000", "110000",
    "110000", "110000", "110000", "110000", "110000", "110000", "110000",
    "110000", "110000", "110101", "110101", "110101", "110101", "110000",
    "110000", "110000", "110101", "110101", "110101", "110101", "110000",
    "110000", "110000", "110101", "110101", "110101", "110101", "110000",
    "110000", "110000", "110101", "110101", "110101", "110101", "110000",
    "110000", "110000", "110000", "110000", "110000", "110000", "110000"]
    #own_grid = ["111111"]*L**2
    #own_grid[L**2//2] = "110000"
    '''
    
    #grid.setup_from_bitcodes(own_grid)
    grid.setup_random()
    
    powers = np.linspace(1.5,1.6,10)
    print(powers)
    moral_score = np.zeros((10,2))
    for i, power in enumerate(powers):
        random.seed(10)
        grid.setup_random()
        print(power)
        game = Game(grid, SimpleUtility(), ReputationManager(gain_base=0.1, loss_base=0.01), asking_style = "distributed", prob_power=power) ## Choose and asking_style between "random", "best" and "distributed"

        evolution = Evolution(game, inverse_copy_prob=60, inverse_mutation_prob=1000, inverse_pardon_prob=200, random_mutation=True)
        evolution.run_interactive(record_data = True, plotting_frequenz=1000)
        evolution.plot_history(i)
        evolution.plot_average_utility(i)
        evolution.plot_average_reputation(i)
    '''
    Sweeper = Analyze_hyper_paramter(
        grid, 
        utility_class=SimpleUtility,
        rep_class=ReputationManager,
        asking_style="random",
        inverse_copy_prob=60,
        inverse_mutation_prob=1000,
        inverse_pardon_prob=200,
        random_mutation=True)

    Sweeper.sweep_rep_loss(np.arange(0.005, 0.105, 0.005, ), rounds=5000, repetitions=5, save_path="plots/sweeps/rep_loss_sweep3.png")
    '''