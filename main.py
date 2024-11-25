import numpy as np
from backend import *

class SimpleUtility(UtilityFunction):
    def calculate(self, action, favor_size):
        """
        Simple utility calculation:
        - Cooperation: Requester gains +favor_size, responder loses -favor_size.
        - Rejection: No utility change.
        """
        if action == "cooperate":
            return favor_size, -favor_size*0.5
        elif action == "reject":
            return 0, 0
        else:  # No action
            return 0, 0

if __name__ == "__main__":
    # Initialize the grid
    L = 7  # Grid size
    N = 1  # Neighborhood radius
    diagonal_neighbors = True  # Include diagonal neighbors

    grid = GameGrid(L, N, diagonal_neighbors)

    # Initialize the game with a simple utility function
    utility_function = SimpleUtility()
    game = Game(grid, utility_function, everyone_can_ask=True)

    game.play_rounds(70)

    analyzer = GameAnalyzer(game)
    analyzer.analyze_strategy_performance()
    analyzer.plot_strategy_grid(show_arrows=False)
    #analyzer.summarize_player(13)

