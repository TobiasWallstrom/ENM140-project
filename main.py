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
        - Cooperation: Requester gains +favor_size, responder loses -favor_size/2.
        - Rejection: No utility change.
        """
        if action == "cooperate":
            return favor_size, -favor_size/2
        elif action == "reject":
            return 0, 0
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


def plot_grid_player_and_neighbors(grid, player_id):
    """ plot the grid and fill the square of the player and its neighbors"""
    fig, ax = plt.subplots()

    # limits of the plot
    grid_size = grid.L
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # let y-axis start at left upper corner and increase downward
    plt.gca().invert_yaxis()

    # Create grid using a mesh
    for x in range(grid_size):
        ax.axhline(x, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x, color='gray', linestyle='-', linewidth=0.5)

    # Color the player-cell
    row, col = divmod(player_id, grid_size)
    ax.add_patch(plt.Rectangle((col, row), 1, 1, color='lightblue'))

    # Color the neighboring cells
    player = grid.players[player_id]
    neighbors = player.neighbors 

    for nb in neighbors:
        row, col = divmod(nb.id, grid_size)
        # Create grid of dots inside the cell
        dot_spacing = 0.1  
        dot_radius = 0.03 
        dot_positions_x = np.arange(col + dot_spacing / 2, col + 1, dot_spacing)
        dot_positions_y = np.arange(row + dot_spacing / 2, row + 1, dot_spacing)
        # Add dots as circles to the plot
        for x in dot_positions_x:
            for y in dot_positions_y:
                circle = plt.Circle((x, y), dot_radius, color='lightgreen', lw=0)
                ax.add_artist(circle)

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    # Set aspect ratio to be equal to ensure square grid cells
    ax.set_aspect('equal')

    plt.show()


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

    #plot_grid_player_and_neighbors(grid, player_id=23)
    
    
    powers = [0.5]
    print(powers)
    for power in powers:
        random.seed(10)
        grid.setup_random()
        print(power)
        game = Game(grid, SimpleUtility(), ReputationManager(gain_base=0.1, loss_base=0.1), asking_style = "distributed", prob_power=power, favor_sizes = [1,3]) ## Choose and asking_style between "random", "best" and "distributed"

        evolution = Evolution(game, inverse_copy_prob=60, inverse_mutation_prob=1000, inverse_pardon_prob=200, random_mutation=True)
        evolution.run_evolution(rounds = 1000)
        evolution.plot_history(power)
        evolution.plot_average_utility(power)
        evolution.plot_average_reputation(power)
    '''
    Sweeper = Analyze_hyper_paramter(
        GameGrid(15, N, strategy_generator_instance, diagonal_neighbors=True), 
        utility_class=SimpleUtility,
        rep_class=ReputationManager,
        asking_style="distributed",
        inverse_copy_prob=60,
        inverse_mutation_prob=1000,
        inverse_pardon_prob=200,
        prob_power = 1.3,
        random_mutation=True)

    #Sweeper.sweep_rep_loss(np.arange(0.005, 0.105, 0.005 ), rounds=5000, repetitions=3, save_path="plots/sweeps/rep_loss_sweep4.png")
    Sweeper.sweep_neighbor_size(np.arange(1, 15, 1), rounds=5000, repetitions=5, save_path="plots/sweeps/neighbor_size_sweep2.png")
    '''