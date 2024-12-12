import numpy as np
import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import copy

class GameAnalyzer:
    def __init__(self, game):
        """
        Initialize the GameAnalyzer with a game instance.
        :param game: The game instance to analyze.
        """
        self.game = game

    def average_utility_per_strategy(self):
        """
        Calculate the average utility for each strategy.
        :return: A dictionary with strategy names as keys and their average utility as values.
        """
        strategy_utilities = defaultdict(list)
        for player in self.game.grid.players:
            strategy_utilities[player.strategy_name].append(player.total_utility)

        # Calculate average utility for each strategy
        return {
            strategy: sum(utilities) / len(utilities) if utilities else 0
            for strategy, utilities in strategy_utilities.items()
        }

    def reputation_distribution(self):
        """
        Calculate the distribution of public reputations.
        :return: A dictionary with counts of each reputation value.
        """
        reputation_counts = defaultdict(int)
        for player in self.game.grid.players:
            reputation_counts[player.public_reputation] += 1
        return dict(reputation_counts)

    def print_report(self):
        """
        Print a detailed report of the game results, including:
        - Average utility per strategy
        - Reputation distribution
        """
        print("Game Analysis Report")
        print("=" * 30)

        # Average utility per strategy
        print("\nAverage Utility Per Strategy:")
        average_utilities = self.average_utility_per_strategy()
        for strategy, avg_utility in average_utilities.items():
            print(f"{strategy}: {avg_utility:.2f}")

        # Reputation distribution
        print("\nReputation Distribution:")
        reputation_dist = self.reputation_distribution()
        for reputation, count in reputation_dist.items():
            print(f"Reputation {reputation}: {count} players")

        print("=" * 30)

class StrategyGenerator:
    def __init__(self, favor_sizes, reputation_values, filter_strategies=True):
        """
        Initialize the StrategyGenerator.
        :param favor_sizes: List of possible favor sizes.
        :param reputation_values: List of possible reputation values.
        :param filter_strategies: Boolean indicating whether to filter unlogical strategies.
        """
        self.favor_sizes = favor_sizes
        self.reputation_values = reputation_values
        self.situations = self._define_situations()
        self.filter_strategies = filter_strategies
        self.strategy_list = []  # List to store all strategies and their colors
        self.strategy_color_map = {}  # Map bitcode to color

    def _define_situations(self):
        situations = []
        for fs in self.favor_sizes:
            situations.append(("ask", fs))
        for fs in self.favor_sizes:
            for r in self.reputation_values:
                situations.append(("help", fs, r))
        return situations

    def _generate_random_color(self):
        """Generate a random color in hexadecimal format."""
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def is_logical(self, decisions):
        """
        Check if a strategy is logical.
        A strategy is logical if:
        - It does not ask for large favors but ignores small favors.
        - It does not help with large favors but ignores small favors.
        - It helps players with a negative reputation only if it helps players with a positive reputation.
        """
        ask_small = decisions.get(("ask", self.favor_sizes[0]), 0)
        ask_large = decisions.get(("ask", self.favor_sizes[1]), 0)

        # Logical check for asking
        if not ask_small and ask_large:  # Asks for large favors but ignores small ones
            return False

        # Logical check for helping
        for r in self.reputation_values:
            help_small = decisions.get(("help", self.favor_sizes[0], r), 0)
            help_large = decisions.get(("help", self.favor_sizes[1], r), 0)
            if not help_small and help_large:  # Helps with large favors but ignores small ones
                return False

        # Logical check for helping if the player has a negative reputation
        for fs in self.favor_sizes:
            help_bad = decisions.get(("help", fs, -1), 0)
            help_good = decisions.get(("help", fs, 1), 0)
            if help_bad and not help_good:
                return False

        return True

    def generate_all_strategies(self):
        """Generate all possible logical strategies and assign unique colors."""
        num_situations = len(self.situations)
        strategies = []

        # Step 1: Generate all strategies
        for decision_vector in itertools.product([0, 1], repeat=num_situations):
            decisions = dict(zip(self.situations, decision_vector))
            if not self.filter_strategies or self.is_logical(decisions):
                strategy = self._create_strategy(decisions)
                strategies.append(strategy)

                # Ensure the strategy has a unique color
                if strategy.bitcode not in self.strategy_color_map:
                    self.strategy_color_map[strategy.bitcode] = self._generate_random_color()

        # Step 2: Assign colors to all strategies
        for strategy in strategies:
            strategy.color = self.strategy_color_map[strategy.bitcode]

        self.strategy_list = strategies
        return strategies

    def _create_strategy(self, decisions):
        """Create a dynamic strategy based on decisions."""
        class DynamicStrategy:
            def __init__(self, decisions, strategy_generator):
                self.decisions = decisions
                self.strategy_generator = strategy_generator
                self.bitcode = "".join(map(str, decisions.values()))
                self.color = None  # This will be assigned in generate_all_strategies
                raw_name = "".join(map(str, self.decisions.values()))
                self.bitcode = raw_name
                self.name = raw_name[:2] + "-" + raw_name[2:]


            def flip_random_bit(self):
                """Flip a random bit and update the strategy to match a predefined one."""
                bit_list = list(self.bitcode)
                tried_indices = set()

                while len(tried_indices) < len(bit_list):
                    # Select a random bit to flip
                    n = random.randint(0, len(bit_list) - 1)
                    if n in tried_indices:
                        continue

                    tried_indices.add(n)

                    # Flip the bit
                    bit_list[n] = '1' if bit_list[n] == '0' else '0'
                    new_bitcode = "".join(bit_list)

                    # Check if the new bitcode exists in the strategy list
                    for strategy in self.strategy_generator.strategy_list:
                        if strategy.bitcode == new_bitcode:
                            # Update all attributes to match the new strategy
                            self.decisions = strategy.decisions
                            self.bitcode = strategy.bitcode
                            self.name = strategy.name  # Synchronize name with the bitcode
                            self.color = strategy.color  # Synchronize color with the bitcode
                            return

                    # Revert the bit if no match is found
                    bit_list[n] = '1' if bit_list[n] == '0' else '0'

                raise ValueError("No matching strategy found after flipping all bits.")

            def ask_for_help(self, player, neighbors):
                favor_size = random.choices([1, 3], weights=[0.5, 0.5])[0]
                if ("ask", favor_size) in self.decisions and self.decisions[("ask", favor_size)] == 1 and neighbors:
                    best_neighbors = [
                        n for n in neighbors if n.public_reputation == max(neighbors, key=lambda n: n.public_reputation).public_reputation
                    ]
                    chosen_neighbor = random.choice(best_neighbors) if len(best_neighbors) > 1 else best_neighbors[0]
                    return {"favor_size": favor_size, "target": chosen_neighbor.id, "action": "ask"}
                return {"favor_size": None, "target": None, "action": "none"}

            def respond_to_help(self, player, requester_id, favor_size):
                return self.decisions.get(("help", favor_size, player.public_reputation), 0) == 1

        return DynamicStrategy(decisions, self)

class Player:
    def __init__(self, player_id, strategy):
        self.id = player_id
        self.strategy = strategy
        self.strategy_name = strategy.name
        self.total_utility = 0
        self.real_reputation = 0.0  # Real reputation as a floating-point value
        self.public_reputation = 0  # Public reputation as a discrete value (-1 or 1)
        self.neighbors = []  # List of neighboring players
        self.recent_utilities = []  # Utilities from the last x rounds
        self.max_recent_rounds = 20  # Maximum number of recent rounds to track

    def update_utility(self, utility_change):
        self.total_utility += utility_change
        self.recent_utilities.append(utility_change)
        # Keep only the last x rounds
        if len(self.recent_utilities) > self.max_recent_rounds:
            self.recent_utilities.pop(0)

    def get_average_utility_per_round(self):
        if len(self.recent_utilities) == 0:
            return 0
        return np.mean(self.recent_utilities)

    def decide_ask_for_help(self):
        return self.strategy.ask_for_help(self, self.neighbors)

    def decide_respond_to_help(self, requester_id, favor_size):
        return self.strategy.respond_to_help(self, requester_id, favor_size)

class GameGrid:
    def __init__(self, L, N, strategy_generator = None, diagonal_neighbors=True):
        self.L = L
        self.N = N
        self.diagonal_neighbors = diagonal_neighbors
        self.strategy_generator = strategy_generator
        self.players = [
            Player(player_id, self.strategy_generator())
            for player_id in range(L * L)
        ]
        self._precompute_neighbors()

    def strategy_generator(self):
        strategy_gen = StrategyGenerator([1, 3], [-1, 1])
        return random.choice(strategy_gen.generate_all_strategies())

    def _precompute_neighbors(self):
        for player in self.players:
            player.neighbors = [self.players[n_id] for n_id in self.get_neighbors(player.id)]

    def shuffle_players(self):
        return random.sample(self.players, len(self.players))

    def get_neighbors(self, player_id):
        x, y = divmod(player_id, self.L)
        neighbors = []
        for i in range(-self.N, self.N + 1):
            for j in range(-self.N, self.N + 1):
                if not self.diagonal_neighbors:
                    if abs(i) + abs(j) > self.N:
                        continue
                else:
                    if max(abs(i), abs(j)) > self.N:
                        continue
                nx = (x + i) % self.L
                ny = (y + j) % self.L
                if (nx, ny) != (x, y):
                    neighbors.append(nx * self.L + ny)
        return neighbors

class Game:
    def __init__(self, grid, utility_function, reputation_manager):
        self.grid = grid
        self.utility_function = utility_function
        self.history = []
        self.reputation_manager = reputation_manager

    def one_round(self):
        players_in_order = self.grid.shuffle_players()
        for player in players_in_order:
            ask_decision = player.decide_ask_for_help()
            if ask_decision["action"] == "ask":
                target = next(p for p in player.neighbors if p.id == ask_decision["target"])
                favor_size = ask_decision["favor_size"]
                response = target.decide_respond_to_help(player.id, favor_size)
                if response:
                    utility_requester, utility_responder = self.utility_function.calculate("cooperate", favor_size)
                    player.update_utility(utility_requester)
                    target.update_utility(utility_responder)
                    self.reputation_manager.update_reputation(player, "accept", favor_size)
                    self.reputation_manager.update_reputation(target, "accept", favor_size)
                else:
                    utility_requester, utility_responder = self.utility_function.calculate("reject", favor_size)
                    player.update_utility(utility_requester)
                    target.update_utility(utility_responder)
                    self.reputation_manager.update_reputation(player, "reject", favor_size)
                    self.reputation_manager.update_reputation(target, "reject", favor_size)

    def play_rounds(self, num_rounds):
        for _ in range(num_rounds):
            self.one_round()

class UtilityFunction:
    def calculate(self, action, favor_size):
        """
        Calculate the utility change for the given action.
        :param action: The action taken ("cooperate", "reject", etc.).
        :param favor_size: The size of the favor involved.
        :return: Tuple (utility_for_requester, utility_for_responder)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class Evolution:
    def __init__(self, game, inverse_copy_prob, inverse_mutation_prob, random_mutation=True):
        """
        Initialize the Evolution class.
        :param game: The Game instance.
        :param inverse_copy_prob: The inverse probability for a mutation to occur.
        :param inverse_mutation_prob: The inverse probability for a mutation.
        :param random_mutation: Boolean to allow random bit flipping during mutation.
        """
        self.game = game
        self.inverse_copy_prob = inverse_copy_prob
        self.inverse_mutation_prob = inverse_mutation_prob
        self.running = False  # Control flag for the evolution process
        if random_mutation:
            self._mutate = self._mutate_both
        else:
            self._mutate = self._mutate_copy

    def _start(self, event):
        """Start the evolution."""
        self.running = True

    def _stop(self, event):
        """Stop the evolution."""
        self.running = False

    def _mutate_copy(self):
        """
        Execute mutation for each player in the grid based on the average utility per round.
        A player does not change their strategy if their average utility
        is equal to or higher than the best neighbor.
        """
        for player in self.game.grid.players:
            if np.random.rand() < 1 / self.inverse_copy_prob:
                # Check if player has neighbors
                if not player.neighbors:
                    continue  # Skip mutation if no neighbors

                # Find the neighbor with the highest average utility per round
                best_neighbor = max(
                    player.neighbors,
                    key=lambda p: p.get_average_utility_per_round()
                )

                # Compare the player's average utility with the best neighbor's
                if player.get_average_utility_per_round() >= best_neighbor.get_average_utility_per_round():
                    continue  # Do not change strategy if player's utility is equal or higher

                # Adopt the strategy of the best-performing neighbor
                player.strategy = copy.deepcopy(best_neighbor.strategy)
                player.strategy_name = best_neighbor.strategy.name

    def _mutate_both(self):
        """
        Perform both copying and random mutation.
        """
        self._mutate_copy()
        for player in self.game.grid.players:
            if np.random.rand() < 1 / self.inverse_mutation_prob:
                player.strategy.flip_random_bit()
                player.strategy_name = player.strategy.name
                print([player2.strategy_name for player2 in self.game.grid.players])
                print([player2.strategy.bitcode for player2 in self.game.grid.players])
                print([player2.strategy.name for player2 in self.game.grid.players])

    def run_interactive(self):
        """
        Run the evolution with a GUI for Start/Stop control.
        """
        plt.ion()  # Enable interactive mode

        # Create a unified mapping for strategies to indices and colors
        strategy_color_map = self.game.grid.players[0].strategy.strategy_generator.strategy_color_map
        strategies = list(strategy_color_map.keys())
        strategy_to_index = {strategy: idx for idx, strategy in enumerate(strategies)}
        cmap = plt.cm.get_cmap("tab20", len(strategies))  # Fixed colormap with unique colors

        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(bottom=0.2)

        # Strategy Grid
        ax1.set_title("Strategy Grid")
        strategy_grid = self._get_strategy_grid(strategy_to_index)
        img = ax1.imshow(strategy_grid, cmap=cmap, interpolation="nearest")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Strategy Overview Table
        ax2.set_title("Strategy Overview")
        ax2.axis("off")
        strategy_table = None

        # Buttons
        ax_start = plt.axes([0.1, 0.05, 0.1, 0.075])
        ax_stop = plt.axes([0.25, 0.05, 0.1, 0.075])
        btn_start = Button(ax_start, "Start")
        btn_stop = Button(ax_stop, "Stop")

        btn_start.on_clicked(self._start)
        btn_stop.on_clicked(self._stop)

        print("Interactive GUI started. Use 'Start' and 'Stop' buttons to control the simulation.")

        while plt.get_fignums():  # Keep running while the figure is open
            if self.running:
                self.game.one_round()
                self._mutate()

                # Update Strategy Grid
                strategy_grid = self._get_strategy_grid(strategy_to_index)
                img.set_data(strategy_grid)

                # Update Strategy Overview Table
                if strategy_table:
                    strategy_table.remove()
                strategy_table = self._update_strategy_table(ax2, strategy_to_index, cmap)

            plt.pause(0.1)  # Allow GUI updates

    def _get_strategy_grid(self, strategy_to_index):
        """
        Return the strategy grid as a 2D numpy array of numerical indices.
        """
        return np.array([
            [strategy_to_index[player.strategy.bitcode] for player in self.game.grid.players]
        ]).reshape(self.game.grid.L, self.game.grid.L)

    def _update_strategy_table(self, ax, strategy_to_index, cmap):
        """
        Update the strategy overview table to display the top 10 strategies
        with corresponding colors and statistics.
        """
        strategy_counts = defaultdict(list)
        for player in self.game.grid.players:
            strategy_counts[player.strategy.bitcode].append(player)

        # Sort strategies by the number of players using them
        strategy_summary = sorted(strategy_counts.items(), key=lambda x: -len(x[1]))

        # Limit to top 10 strategies
        top_strategies = strategy_summary[:10]

        # Add placeholders if there are fewer than 10 strategies
        while len(top_strategies) < 10:
            top_strategies.append(("None", []))

        # Extract data for the table
        table_data = []
        for strategy, players in top_strategies:
            percentage = len(players) / len(self.game.grid.players) * 100 if players else 0
            mean_utility = np.mean([p.get_average_utility_per_round() for p in players]) if players else 0
            std_utility = np.std([p.get_average_utility_per_round() for p in players]) if players else 0
            mean_reputation = np.mean([p.real_reputation for p in players]) if players else 0
            std_reputation = np.std([p.real_reputation for p in players]) if players else 0

            # Get color index
            color = cmap(strategy_to_index.get(strategy, 0) / len(strategy_to_index)) if strategy != "None" else "#FFFFFF"

            # Add row to the table
            table_data.append([
                f"{strategy}",
                f"{percentage:.2f}%",
                f"{mean_utility:.2f} ± {std_utility:.2f}",
                f"{mean_reputation:.2f} ± {std_reputation:.2f}",
                color
            ])

        # Create or update the table
        table = ax.table(
            cellText=[row[:-1] for row in table_data],  # Exclude the color column for text
            colLabels=["Strategy", "Percentage", "Mean Utility", "Mean Rep"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Set font size for better readability
        table.scale(1.0, 1.0)  # Adjust table size

        # Add colors to the cells in the first column
        for row_idx, row in enumerate(table_data):
            color = row[-1]
            cell = table[row_idx + 1, 0]  # Offset for header row
            cell.set_facecolor(color)

        return table
