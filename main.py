import numpy as np
from backend import *

def testGrid():
    # Example usage:
    L = 6  # Size of the grid (LxL)
    N = 1  # Maximum neighborhood radius

    game = GameGrid(L, N)
    game.display_grid()  # Displays the grid in 2D format

    # Example: Find neighbors for a specific player by ID using both methods
    player_id = 8  # Player ID
    neighbors = game.get_neighbors(player_id)

    print(f"Neighbors of player {player_id}: {neighbors}")

if __name__ == "__main__":
    # Initialize the grid
    L = 3  # Grid size
    N = 1  # Neighborhood radius
    diagonal_neighbors = True  # Include diagonal neighbors

    grid = GameGrid(L, N, diagonal_neighbors)

    # Initialize the game with a simple utility function
    utility_function = SimpleUtility()
    game = Game(grid, utility_function)

    game.play_rounds(3)

    #game.display_total_results()  # Display the total results
    game.summarize_player(2)
    game.display_utility_grid()

    print(game.grid.players[2].interaction_history)
