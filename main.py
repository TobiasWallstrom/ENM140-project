import numpy as np

class GameGrid:
    def __init__(self, L, N):
        """
        Initializes the game grid with periodic boundary conditions.
        :param L: Size of the grid (LxL)
        :param N: Neighborhood radius
        """
        self.L = L  # Size of the grid
        self.N = N  # Neighborhood radius
        self.players = np.arange(L * L)  # 1D array representing player IDs

    def get_neighbors_taxigrid(self, player_id):
        """
        Finds the neighbors of a player using Manhattan distance.
        Diagonal fields are not considered direct neighbors.
        :param player_id: ID of the player
        :return: List of IDs of neighboring players
        """
        x, y = divmod(player_id, self.L)  # Convert player ID to grid coordinates

        neighbors = []
        for i in range(-self.N, self.N + 1):
            for j in range(-self.N, self.N + 1):
                # Exclude diagonal neighbors that are too far
                if abs(i) + abs(j) > self.N:
                    continue
                # Periodic boundary conditions
                nx = (x + i) % self.L
                ny = (y + j) % self.L
                # Exclude the player itself
                if (nx, ny) != (x, y):
                    neighbor_id = nx * self.L + ny
                    neighbors.append(neighbor_id)
        return neighbors

    def get_neighbors_diagonal(self, player_id):
        """
        Finds the neighbors of a player considering diagonal fields as direct neighbors.
        :param player_id: ID of the player
        :return: List of IDs of neighboring players
        """
        x, y = divmod(player_id, self.L)  # Convert player ID to grid coordinates

        neighbors = []
        for i in range(-self.N, self.N + 1):
            for j in range(-self.N, self.N + 1):
                # Exclude distances greater than N (including diagonals)
                if max(abs(i), abs(j)) > self.N:
                    continue
                # Periodic boundary conditions
                nx = (x + i) % self.L
                ny = (y + j) % self.L
                # Exclude the player itself
                if (nx, ny) != (x, y):
                    neighbor_id = nx * self.L + ny
                    neighbors.append(neighbor_id)
        return neighbors

    def display_grid(self):
        """
        Displays the grid with player IDs in 2D format for visualization.
        """
        grid = self.players.reshape(self.L, self.L)
        print(grid)

    def get_grid(self):
        """
        Returns the current grid as a 2D array for external visualization.
        """
        return self.players.reshape(self.L, self.L)


# Example usage:
L = 6  # Size of the grid (5x5)
N = 1  # Maximum neighborhood radius

game = GameGrid(L, N)
game.display_grid()  # Displays the grid in 2D format

# Example: Find neighbors for a specific player by ID using both methods
player_id = 8  # Player ID
taxi_neighbors = game.get_neighbors_taxigrid(player_id)
diagonal_neighbors = game.get_neighbors_diagonal(player_id)

print(f"Taxigrid neighbors of player {player_id}: {taxi_neighbors}")
print(f"Diagonal neighbors of player {player_id}: {diagonal_neighbors}")
