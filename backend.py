import numpy as np
import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import copy
from PIL import Image
from PIL.ExifTags import TAGS

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

    def get_avrage_moral_score(self):
        # calculate the moral score with std of the total game weighted by the number of players with the strategy
        moral_scores = []
        for player in self.game.grid.players:
            moral_scores.append(player.strategy.moral_score)
        return np.mean(moral_scores), np.std(moral_scores)

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
        if not self.strategy_list:  # Vermeide doppelte Erstellung
            num_situations = len(self.situations)
            strategies = []

            for decision_vector in itertools.product([0, 1], repeat=num_situations):
                decisions = dict(zip(self.situations, decision_vector))
                if not self.filter_strategies or self.is_logical(decisions):
                    strategy = self._create_strategy(decisions)
                    strategies.append(strategy)

                    # Farbzuweisung
                    if strategy.bitcode not in self.strategy_color_map:
                        self.strategy_color_map[strategy.bitcode] = self._generate_random_color()

            # Farben synchronisieren
            for strategy in strategies:
                strategy.color = self.strategy_color_map[strategy.bitcode]

            self.strategy_list = strategies  # Strategien speichern

        return self.strategy_list

    def _create_strategy(self, decisions):
        """Create a dynamic strategy based on decisions."""
        super_favor_sizes = self.favor_sizes
        class DynamicStrategy:
            def __init__(self, decisions, strategy_generator):
                self.decisions = decisions
                self.strategy_generator = strategy_generator
                self.bitcode = "".join(map(str, decisions.values()))
                self.color = None  # This will be assigned in generate_all_strategies
                raw_name = "".join(map(str, self.decisions.values()))
                self.bitcode = raw_name
                self.name = raw_name[:2] + "-" + raw_name[2:]
                self.moral_score = raw_name[2:].count("1") #form 0 to 4


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
                            self.moral_score = strategy.moral_score
                            return

                    # Revert the bit if no match is found
                    bit_list[n] = '1' if bit_list[n] == '0' else '0'

                raise ValueError("No matching strategy found after flipping all bits.")

            def ask_for_help(self, player, neighbors, asking_style="random", prob_power = 1):
                favor_size = random.choices(super_favor_sizes, weights=[0.5, 0.5])[0]
                if ("ask", favor_size) in self.decisions and self.decisions[("ask", favor_size)] == 1 and neighbors:
                    if asking_style == "random":
                        chosen_neighbor = random.choice(neighbors)
                    
                    if asking_style == "best":
                        best_neighbors = [
                            n for n in neighbors if n.public_reputation == max(neighbors, key=lambda n: n.public_reputation).public_reputation
                        ]
                        chosen_neighbor = random.choice(best_neighbors) if len(best_neighbors) > 1 else best_neighbors[0]

                    if asking_style == "distributed":
                        neighbors_rep = [(n.public_reputation + 2)**prob_power for n in neighbors]
                        total_rep = sum(neighbors_rep)
                        weights = [x/total_rep for x in neighbors_rep]
                        chosen_neighbor = np.random.choice(neighbors, 1, p=weights)[0]

                    
                    return {"favor_size": favor_size, "target": chosen_neighbor.id, "action": "ask"}
                return {"favor_size": None, "target": None, "action": "none"}

            def respond_to_help(self, player, requester_id, favor_size):
                requester = next(p for p in player.neighbors if p.id == requester_id)
                anwser = self.decisions.get(("help", favor_size, requester.public_reputation), 0) == 1
                if not anwser:
                    #print("hilftnciht#######################################################################")
                    pass
                return anwser

        return DynamicStrategy(decisions, self)

class Player:
    def __init__(self, player_id, strategy):
        self.id = player_id
        self.strategy = strategy
        self.strategy_name = strategy.name
        self.total_utility = 0
        self.real_reputation = 0.5  # Real reputation as a floating-point value
        self.public_reputation = 1  # Public reputation as a discrete value (-1 or 1)
        self.neighbors = []  # List of neighboring players
        self.recent_utilities = []  # Utilities from the last x rounds
        self.max_recent_rounds = 20  # Maximum number of recent rounds to track
        self.all_utilities = []

    def update_utility(self, utility_change):
        self.total_utility += utility_change
        self.recent_utilities.append(utility_change)
        self.all_utilities.append(utility_change)

        # Keep only the last x rounds
        if len(self.recent_utilities) > self.max_recent_rounds:
            self.recent_utilities.pop(0)

    def update_reputation(self, reputation_change, max_reputation=1, min_reputation=-1):
        self.real_reputation = min(max_reputation, max(min_reputation, self.real_reputation + reputation_change))
        self.public_reputation = max_reputation if self.real_reputation >= 0 else min_reputation

    def get_average_utility_per_round(self):
        if len(self.recent_utilities) == 0:
            return 0
        return np.mean(self.recent_utilities)*2 # multiply by 2 to get the average utility per round instead of per favor_change
    
    def decide_ask_for_help(self, asking_style, prob_power):
        return self.strategy.ask_for_help(self, self.neighbors, asking_style, prob_power)

    def decide_respond_to_help(self, requester_id, favor_size):
        return self.strategy.respond_to_help(self, requester_id, favor_size)
    
    def pardon(self, pardon_size = 0.1):
        if self.real_reputation < (1-pardon_size):
            self.real_reputation += pardon_size

class GameGrid:
    def __init__(self, L, N, strategy_generator_instance, diagonal_neighbors=True):
        """
        Initialize the GameGrid.
        :param L: Size of the grid (L x L).
        :param N: Neighborhood radius.
        :param strategy_generator_instance: An instance of StrategyGenerator to generate or interpret strategies.
        :param diagonal_neighbors: Boolean to include diagonal neighbors.
        """
        self.L = L
        self.N = N
        self.diagonal_neighbors = diagonal_neighbors
        self.strategy_generator_instance = strategy_generator_instance
        self.players = []
        self.setup_random()
        self._precompute_neighbors()

    def setup_random(self):
        """Randomly assign strategies to all players using the strategy generator."""
        self.players = []
        for player_id in range(self.L * self.L):
            strategy = copy.deepcopy(random.choice(self.strategy_generator_instance.generate_all_strategies()))
            self.players.append(Player(player_id, strategy))
        self._precompute_neighbors()

    def setup_from_bitcodes(self, bitcodes):
        """
        Set up the GameGrid using a list of bitcodes.
        :param bitcodes: List of bitcodes to assign to players.
        """
        if len(bitcodes) != self.L * self.L:
            raise ValueError("The number of bitcodes must match the grid size (L x L).")

        # Map bitcodes to strategies
        strategies = {strategy.bitcode: strategy for strategy in self.strategy_generator_instance.generate_all_strategies()}

        self.players = []
        for player_id, bitcode in enumerate(bitcodes):
            if bitcode not in strategies:
                raise ValueError(f"Invalid bitcode '{bitcode}' provided.")
            strategy = copy.deepcopy(strategies[bitcode])
            self.players.append(Player(player_id, strategy))

        self._precompute_neighbors()

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

    def change_L(self, new_L):
        self.L = new_L
        self.setup_random()
        self._precompute_neighbors()

    def change_N(self, new_N):
        self.N = new_N
        self.setup_random()
        self._precompute_neighbors()    

class Game:
    def __init__(self, grid, utility_function, reputation_manager, asking_style, prob_power):
        self.grid = grid
        self.utility_function = utility_function
        self.reputation_manager = reputation_manager
        self.asking_style = asking_style
        self.prob_power = prob_power

    def one_round(self):
        players_in_order = self.grid.shuffle_players()
        for player in players_in_order:
            ask_decision = player.decide_ask_for_help(self.asking_style, self.prob_power)
            if ask_decision["action"] == "ask":
                target = next(p for p in player.neighbors if p.id == ask_decision["target"])
                favor_size = ask_decision["favor_size"]
                response = target.decide_respond_to_help(player.id, favor_size)
                if response:
                    utility_requester, utility_responder = self.utility_function.calculate("cooperate", favor_size)
                    player.update_utility(utility_requester)
                    target.update_utility(utility_responder)
                    self.reputation_manager.update_reputation(player, target, "accept", favor_size)
                else:
                    utility_requester, utility_responder = self.utility_function.calculate("reject", favor_size)
                    player.update_utility(utility_requester)
                    target.update_utility(utility_responder)
                    #self.reputation_manager.update_reputation(player, "reject", favor_size)
                    self.reputation_manager.update_reputation(player, target, "reject", favor_size)

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
    def __init__(self, game, inverse_copy_prob, inverse_mutation_prob, inverse_pardon_prob, random_mutation=True):
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
        self.inverse_pardon_prob = inverse_pardon_prob
        self.running = False  #Control flag for the evolution process
        self.history = []
        if random_mutation:
            self._mutate = self._mutate_both
        else:
            self._mutate = self._mutate_copy
        self.iteration = 0

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

    def _pardon(self):
        nPlayers = len(self.game.grid.players)
        arg_pardon = np.random.rand(nPlayers)
        indicies = np.where(arg_pardon < 1/self.inverse_pardon_prob)[0]
        for i in indicies:
            self.game.grid.players[i].pardon()

    def _mutate_both(self):
        """
        Perform both copying and random mutation.
        """
        self._mutate_copy()
        for player in self.game.grid.players:
            if np.random.rand() < 1 / self.inverse_mutation_prob:
                player.strategy.flip_random_bit()
                player.strategy_name = player.strategy.name

    def run_interactive(self, record_data = True, plotting_frequenz = 100):
        """
        Run the evolution with a GUI for Start/Stop control.
        """
        plt.ion()  # Enable interactive mode

        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(bottom=0.2)

        # Strategy Grid
        ax1.set_title("Strategy Grid")
        strategy_grid = self._get_strategy_grid()
        img = ax1.imshow(strategy_grid, interpolation="nearest")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Strategy Overview Table
        ax2.set_title("Strategy Overview")
        ax2.axis("off")
        self.strategy_table = self.initialize_strategy_table(ax2)
        self.update_strategy_table(self.strategy_table)

        # Iteration Counter
        #iteration = 0
        iteration_text = fig.text(0.5, 0.05, f"Iteration: {self.iteration}", ha='center', va='center', fontsize=12)

        # Buttons
        ax_start = plt.axes([0.1, 0.05, 0.1, 0.075])
        ax_stop = plt.axes([0.25, 0.05, 0.1, 0.075])
        btn_start = Button(ax_start, "Start")
        btn_stop = Button(ax_stop, "Stop")

        btn_start.on_clicked(self._start)
        btn_stop.on_clicked(self._stop)
        plt.show()

        print("Interactive GUI started. Use 'Start' and 'Stop' buttons to control the simulation.")

        while plt.get_fignums():  # Keep running while the figure is open
            if self.running:
                self.run_evolution(1, record_data)

                # Update Strategy Grid
                strategy_grid = self._get_strategy_grid()
                img.set_data(strategy_grid)

                # Update Strategy Overview Table
                self.update_strategy_table(self.strategy_table)
                # Update Iteration Counter
                iteration_text.set_text(f"Iteration: {self.iteration}")

            if self.iteration == 1 or self.iteration % plotting_frequenz == 0:   
                plt.pause(0.01)  # Allow GUI updates'
        
        plt.close(fig)

    def run_evolution(self, rounds, record_data = True, print_rounds = True):
        for _ in range(rounds):
            if self.iteration % 250 == 0 and print_rounds:
                print(f"Round: {self.iteration}")
            self.game.one_round()
            self._mutate()
            self._pardon()
            self.iteration += 1
            if record_data:
                self._record_history(self.iteration)

    def _get_strategy_grid(self):
        """
        Return the strategy grid as a 2D numpy array of RGB color values.
        """
        L = self.game.grid.L
        # Convert colors to RGB tuples
        color_grid = [
            [
                self._hex_to_rgb(player.strategy.color) for player in self.game.grid.players[i * L:(i + 1) * L]
            ]
            for i in range(L)
        ]
        return np.array(color_grid)

    def _hex_to_rgb(self, hex_color):
        """
        Convert a hexadecimal color to an RGB tuple.
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def update_strategy_table(self, table):
        """
        Update the strategy overview table with current data.
        """
        strategy_counts = defaultdict(list)
        for player in self.game.grid.players:
            strategy_counts[player.strategy.bitcode].append(player)

        strategy_summary = sorted(strategy_counts.items(), key=lambda x: -len(x[1]))
        top_strategies = strategy_summary[:10]

        while len(top_strategies) < 10:
            top_strategies.append(("None", []))

        # Track the maximum values for each column
        max_values = {"Percentage": 0, "Mean Utility": float('-inf'), "Mean Rep": float('-inf')}
        max_indices = {"Percentage": -1, "Mean Utility": -1, "Mean Rep": -1}

        for row_idx, (strategy, players) in enumerate(top_strategies):
            percentage = len(players) / len(self.game.grid.players) * 100 if players else 0
            mean_utility = np.mean([p.get_average_utility_per_round() for p in players]) if players else 0
            mean_reputation = np.mean([p.real_reputation for p in players]) if players else 0
            color = players[0].strategy.color if players else "#FFFFFF"  # Take the color of the first player

            # Update maximum values and their indices
            if percentage > max_values["Percentage"]:
                max_values["Percentage"] = percentage
                max_indices["Percentage"] = row_idx

            if mean_utility > max_values["Mean Utility"]:
                max_values["Mean Utility"] = mean_utility
                max_indices["Mean Utility"] = row_idx

            if mean_reputation > max_values["Mean Rep"]:
                max_values["Mean Rep"] = mean_reputation
                max_indices["Mean Rep"] = row_idx

            # Update the table row
            table[row_idx + 1, 0].get_text().set_text(f"{strategy}")
            table[row_idx + 1, 1].get_text().set_text(f"{percentage:.2f}%")
            table[row_idx + 1, 2].get_text().set_text(f"{mean_utility:.2f}")
            table[row_idx + 1, 3].get_text().set_text(f"{mean_reputation:.2f}")

            # Set the color of the first column (strategy column)
            table[row_idx + 1, 0].set_facecolor(color)

            # Remove bold formatting for all rows first
            for col_idx in range(1, 4):  # Exclude the first column
                table[row_idx + 1, col_idx].get_text().set_fontweight("normal")

        # Make the highest values in each column bold
        for col_name, col_idx in zip(["Percentage", "Mean Utility", "Mean Rep"], [1, 2, 3]):
            if max_indices[col_name] >= 0:
                table[max_indices[col_name] + 1, col_idx].get_text().set_fontweight("bold")

    def initialize_strategy_table(self, ax):
        """
        Initialize the strategy overview table with empty data.
        """
        # Create empty placeholder data
        placeholder_data = [["" for _ in range(4)] for _ in range(10)]
        col_labels = ["Strategy", "Percentage", "Mean Utility", "Mean Rep"]

        # Create the table
        table = ax.table(
            cellText=placeholder_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.0)

        # Make header row bold
        for col_idx in range(len(col_labels)):
            cell = table[0, col_idx]
            cell.get_text().set_fontweight("bold")  # Set header text to bold

        return table

    def plot_history(self, power):
        """
        Plot the recorded history of strategies over iterations.
        """
        if not self.history:
            print("No history to plot.")
            return

        # Prepare data for plotting
        iterations = [entry["iteration"] for entry in self.history]
        strategy_data = {}
        
        # Gather all unique strategies across iterations
        for entry in self.history:
            for strat in entry["strategies"]:
                if strat["bitcode"] not in strategy_data:
                    strategy_data[strat["bitcode"]] = {"percentages": [], "utilities": [], "reputations": []}

        # Populate the data for each strategy
        for entry in self.history:
            recorded_strategies = {s["bitcode"]: s for s in entry["strategies"]}
            for bitcode in strategy_data.keys():
                if bitcode in recorded_strategies:
                    strategy_data[bitcode]["percentages"].append(recorded_strategies[bitcode]["percentage"])
                    strategy_data[bitcode]["utilities"].append(recorded_strategies[bitcode]["mean_utility"])
                    strategy_data[bitcode]["reputations"].append(recorded_strategies[bitcode]["mean_reputation"])
                else:
                    strategy_data[bitcode]["percentages"].append(0)
                    strategy_data[bitcode]["utilities"].append(0)
                    strategy_data[bitcode]["reputations"].append(0)

        # Sort strategies by their final percentage in descending order
        sorted_strategies = sorted(strategy_data.items(), key=lambda x: x[1]["percentages"][-1], reverse=True)

        # Plot strategy percentages
        plt.figure(figsize=(12, 6))
        for bitcode, data in sorted_strategies:
            plt.plot(iterations, data["percentages"], label=f"Strategy {bitcode}")
        plt.title("Strategy Distribution Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Percentage of Players")
        plt.legend(title="Strategies (sorted by final percentage)",
                   loc="upper left",  # Change this to other options if needed
                    bbox_to_anchor=(1.05, 1),  # Position it outside the plot
                    borderaxespad=0.  ) # Add padding
        plt.grid()
        plt.tight_layout()

        # Show or save the plot
        plt.savefig(str(power)+".png")
        plt.close()

    def plot_average_utility(self, iteration):
        """
        Plot the average utility of all players over time.
        """
        if not self.history:
            print("No history to plot.")
            return

        # Prepare data for plotting
        iterations = [entry["iteration"] for entry in self.history]
        average_utilities = []

        for entry in self.history:
            # Calculate the overall average utility across all strategies for this iteration
            total_utility = sum(
                strat["mean_utility"] * (strat["percentage"] / 100) for strat in entry["strategies"]
            )
            average_utilities.append(total_utility)

        # Plot the average utility over time
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, average_utilities, linestyle="-", color="blue")
        plt.title("Average Utility of All Players Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Average Utility")
        plt.grid()
        plt.tight_layout()

        # Save or show the plot
        plt.savefig(str(iteration) + "utility.png")
        print("Plot saved as 'average_utility_over_time.png'.")
        plt.close()
    
    def plot_average_reputation(self, iteration):
        """
        Plot the average reputation of all players over time.
        """
        if not self.history:
            print("No history to plot.")
            return

        # Prepare data for plotting
        iterations = [entry["iteration"] for entry in self.history]
        average_reputations = []

        for entry in self.history:
            # Calculate the overall average reputation across all strategies for this iteration
            total_reputation = sum(
                strat["mean_reputation"] * (strat["percentage"] / 100) for strat in entry["strategies"]
            )
            average_reputations.append(total_reputation)

        # Plot the average reputation over time
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, average_reputations, linestyle="-", color="green")
        plt.title("Average Reputation of All Players Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reputation")
        plt.grid()
        plt.tight_layout()

        # Save or show the plot
        plt.savefig(str(iteration)+"reputation.png")
        print("Plot saved as 'average_reputation_over_time.png'.")
        plt.close()

    def _record_history(self, iteration):
        """
        Record the current state of the strategies and statistics.
        """
        strategy_counts = defaultdict(list)
        for player in self.game.grid.players:
            strategy_counts[player.strategy.bitcode].append(player)

        # Summarize data
        history_entry = {
            "iteration": iteration,
            "strategies": []
        }
        for bitcode, players in strategy_counts.items():
            percentage = len(players) / len(self.game.grid.players) * 100
            mean_utility = np.mean([p.get_average_utility_per_round() for p in players])
            mean_reputation = np.mean([p.real_reputation for p in players])
            history_entry["strategies"].append({
                "bitcode": bitcode,
                "percentage": percentage,
                "mean_utility": mean_utility,
                "mean_reputation": mean_reputation
            })

        self.history.append(history_entry)

class Analyze_hyper_paramter:
    def __init__(self, grid, utility_class, rep_class, asking_style, inverse_copy_prob, inverse_mutation_prob, inverse_pardon_prob, prob_power = 1, random_mutation=True):
        self.grid = grid
        self.utility_class = utility_class
        self.rep_class = rep_class
        self.asking_style = asking_style
        self.inverse_copy_prob = inverse_copy_prob
        self.inverse_mutation_prob = inverse_mutation_prob
        self.inverse_pardon_prob = inverse_pardon_prob
        self.random_mutation = random_mutation
        self.prob_power = prob_power

    def sweep_rep_loss(self, rep_loss_values, rounds=5000, plot_results=True, repetitions=3, save_path="plots/sweeps/plot.png"):
        """
        Sweep the reputation loss base values, analyze the results, and save the plot with EXIF metadata.
        """
        results = {}
        for rep_loss in rep_loss_values:
            print(f"Round {np.where(rep_loss_values == rep_loss)[0][0]+1} of {len(rep_loss_values)}")
            moral_scores = []
            for _ in range(repetitions):
                rep_manager = self.rep_class(loss_base=rep_loss)
                game = Game(self.grid, self.utility_class(), rep_manager, self.asking_style, self.prob_power)
                evolution = Evolution(game, self.inverse_copy_prob, self.inverse_mutation_prob, self.inverse_pardon_prob, self.random_mutation)
                evolution.run_evolution(rounds, True, False)
                analyzer = GameAnalyzer(game)
                avg_score, std_score = analyzer.get_avrage_moral_score()
                moral_scores.append(avg_score)

            # Calculate mean and std across repetitions
            results[rep_loss] = (np.mean(moral_scores), np.std(moral_scores))

        if plot_results:
            rep_losses = list(results.keys())
            moral_scores = [score[0] for score in results.values()]
            moral_score_stds = [score[1] for score in results.values()]

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.errorbar(rep_losses, moral_scores, yerr=moral_score_stds, fmt='-o', capsize=5, label='Average Moral Score')
            plt.fill_between(rep_losses, np.array(moral_scores) - np.array(moral_score_stds), np.array(moral_scores) + np.array(moral_score_stds), alpha=0.2)
            plt.xlabel('Reputation Loss Base Value')
            plt.ylabel('Average Moral Score')
            plt.title('Average Moral Score vs. Reputation Loss Base Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

            # Save EXIF metadata
            metadata = {
                "Grid Size": self.grid.L,
                "Neighborhood Radius": self.grid.N,
                "Inverse Copy Probability": self.inverse_copy_prob,
                "Inverse Mutation Probability": self.inverse_mutation_prob,
                "Inverse Pardon Probability": self.inverse_pardon_prob,
                "Repetitions": repetitions,
                "Rounds Per Simulation": rounds,
            }

            # Open the plot and add EXIF metadata
            img = Image.open(save_path)
            exif_data = {
                TAGS.get(tag, tag): val
                for tag, val in img.info.get("exif", {}).items()
            }
            for key, value in metadata.items():
                exif_data[key] = value

            # Save the updated image with EXIF metadata
            img.save(save_path, exif=img.info.get("exif"))

        return results

        
    