from backend import *

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


    
L = 7  # Grid size
N = 2   # Neighborhood radius

strategy_generator_instance = StrategyGenerator(
    favor_sizes=[1, 3],
    reputation_values=[-1, 1]
)
grid = GameGrid(L, N, strategy_generator_instance, diagonal_neighbors=False)
id = 23


plot_grid_player_and_neighbors(grid,id)