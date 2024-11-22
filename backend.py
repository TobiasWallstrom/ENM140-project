import numpy as np
import random
import importlib
import inspect
from strategies import *
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class Game:
    def __init__(self, grid, utility_function, everyone_can_ask=True):
        """
        Initialize the game with a grid and a utility function.
        :param grid: An instance of GameGrid containing players and neighbors.
        :param utility_function: An instance of UtilityFunction to calculate utility changes.
        """
        self.grid = grid  # GameGrid instance
        self.utility_function = utility_function  # Utility function instance
        self.history = []  # List to store interaction history for each round
        self.everyone_can_ask = everyone_can_ask

    def one_round(self):
        """
        Play one round of the game. Each player can ask for help or respond to requests.
        """
        players_in_order = self.grid.shuffle_players()  # Randomized player order
        interacted_players = set()  # Track players who have already interacted
        round_history = []  # Store events for this round

        for player in players_in_order:
            # Skip if the player has already interacted
            if player.id in interacted_players:
                continue

            # Add player to the list of interacted players
            interacted_players.add(player.id)

            # Get the player's neighbors
            neighbors = self.grid.get_neighbors(player.id)

            # Ask the strategy if the player wants to ask for help
            ask_decision = player.decide_ask_for_help(neighbors)
            if ask_decision["action"] == "ask":
                target_id = ask_decision["target"]
                favor_size = ask_decision["favor_size"]

                # Find the target player
                target_player = next(p for p in self.grid.players if p.id == target_id)

                # Skip if the target player has already interacted
                if target_id in interacted_players:
                    player.record_interaction(target_id, "busy")
                    target_player.record_interaction(player.id, "asked for help; was busy")
                    round_history.append({
                        "initiator": player.id,
                        "target": target_id,
                        "action": "ask",
                        "result": "busy",
                        "favor_size": favor_size
                    })
                    continue

                # Ask the target player if they will help
                response = target_player.decide_respond_to_help(player.id, favor_size)
                
                if response:  # Cooperation
                    # Calculate utility changes
                    utility_requester, utility_responder = self.utility_function.calculate("cooperate", favor_size)

                    # Update utilities for both players
                    player.update_utility(utility_requester)
                    target_player.update_utility(utility_responder)

                    # Record interactions
                    player.record_interaction(target_player.id, "got favor")
                    target_player.record_interaction(player.id, "helped with favor")

                    # Mark the target player as interacted if "everyone_can_ask" is False
                    if not self.everyone_can_ask:
                        interacted_players.add(target_player.id)

                    # Add to round history
                    round_history.append({
                        "initiator": player.id,
                        "target": target_id,
                        "action": "ask",
                        "result": "cooperate",
                        "favor_size": favor_size
                    })
                else:  # Rejection
                    # Calculate utility changes for rejection
                    utility_requester, utility_responder = self.utility_function.calculate("reject", favor_size)

                    # Update utilities (typically no change, but flexible)
                    player.update_utility(utility_requester)
                    target_player.update_utility(utility_responder)

                    # Record the rejection
                    player.record_interaction(target_player.id, "got rejected")
                    target_player.record_interaction(player.id, "asked for favor; did not help")

                    # Add to round history
                    round_history.append({
                        "initiator": player.id,
                        "target": target_id,
                        "action": "ask",
                        "result": "reject",
                        "favor_size": favor_size
                    })
            else:  # No action
                player.record_interaction(None, "no_action")
                round_history.append({
                    "initiator": player.id,
                    "target": None,
                    "action": "none",
                    "result": "no_action",
                    "favor_size": None
                })

        # Append this round's history to the game history
        self.history.append(round_history)

    def play_rounds(self, num_rounds):
        """
        Play multiple rounds of the game.
        :param num_rounds: Number of rounds to play.
        """
        for round_number in range(1, num_rounds + 1):
            #print(f"\n--- Round {round_number} ---")
            self.one_round()

class GameAnalyzer:
    def __init__(self, game):
        """
        Initialize the GameAnalyzer with a Game instance.
        :param game: The Game instance to analyze.
        """
        self.game = game
        self.cooperation_counts = None  # Strategy cooperation matrix
        self.strategy_partners = None  # Most frequent cooperation partners
        self.strategy_utilities = None  # Strategy utility data

    def analyze_strategy_performance(self):
        """
        Analyze and display strategy performance and cooperation dynamics.
        """
        # Collect utilities by strategy
        strategy_utilities = defaultdict(list)
        cooperation_counts = defaultdict(lambda: defaultdict(int))
        strategy_partners = defaultdict(lambda: defaultdict(int))

        for player in self.game.grid.players:
            strategy_utilities[player.strategy_name].append(player.total_utility)

        for round_events in self.game.history:
            for event in round_events:
                if event["result"] == "cooperate":
                    initiator = next(p for p in self.game.grid.players if p.id == event["initiator"])
                    target = next(p for p in self.game.grid.players if p.id == event["target"])
                    cooperation_counts[initiator.strategy_name][target.strategy_name] += 1
                    strategy_partners[initiator.id][target.id] += 1

        # Save results for later use
        self.strategy_utilities = strategy_utilities
        self.cooperation_counts = cooperation_counts
        self.strategy_partners = strategy_partners

        # Display results
        print("\n--- Strategy Performance Analysis ---")
        for strategy, utilities in strategy_utilities.items():
            avg_utility = np.mean(utilities)
            std_utility = np.std(utilities)
            print(f"Strategy: {strategy}, Average Utility: {avg_utility:.2f}, Std Dev: {std_utility:.2f}")

        print(f"\n--- Number of Unique Strategies: {len(strategy_utilities)} ---")

        print("\n--- Strategy Cooperation Matrix ---")
        strategies = sorted(cooperation_counts.keys())
        header = " " * 15 + " ".join(f"{s:15}" for s in strategies)
        print(header)
        for strategy1 in strategies:
            row = f"{strategy1:15}"
            for strategy2 in strategies:
                row += f"{cooperation_counts[strategy1][strategy2]:15}"
            print(row)

    def display_game_history(self):
        """
        Display the history of all interactions across all rounds.
        """
        print("\n--- Game History ---")
        for round_num, round_events in enumerate(self.game.history, start=1):
            print(f"\n--- Round {round_num} ---")
            if not round_events:
                print("No interactions in this round.")
            for event in round_events:
                initiator = event.get("initiator", None)
                target = event.get("target", None)
                action = event.get("action", "none")
                result = event.get("result", "no_action")
                favor_size = event.get("favor_size", None)

                print(f"Player {initiator} -> Player {target} | Action: {action} | "
                      f"Result: {result} | Favor Size: {favor_size}")

    def summarize_player(self, player_id):
        """
        Summarize a specific player's history and total utility.
        :param player_id: ID of the player to summarize.
        """
        player = next(p for p in self.game.grid.players if p.id == player_id)
        total_utility = player.total_utility

        times_asked = 0
        successful_requests = 0
        rejected_requests = 0

        print(f"\n--- Summary for Player {player_id} ---")
        for round_num, round_events in enumerate(self.game.history, start=1):
            player_events = [event for event in round_events if event["initiator"] == player_id or event["target"] == player_id]
            if not player_events:
                print(f"Round {round_num}: No interactions.")
            else:
                for event in player_events:
                    action = event.get("action", "none")
                    result = event.get("result", "no_action")
                    target = event.get("target", None)
                    favor_size = event.get("favor_size", None)
                    initiator = event.get("initiator", None)

                    if event["target"] == player_id:
                        times_asked += 1
                        if result == "cooperate":
                            successful_requests += 1
                        elif result == "reject":
                            rejected_requests += 1
                        print(f"Round {round_num}: Asked by Player {initiator}, Action = {action}, "
                              f"Result = {result}, Favor Size = {favor_size}")
                    elif event["initiator"] == player_id:
                        print(f"Round {round_num}: Action = {action}, Target = {target}, "
                              f"Result = {result}, Favor Size = {favor_size}")

        print(f"Total Utility: {total_utility}")
        print(f"Times Asked for Help: {times_asked} (Cooperated: {successful_requests}, Rejected: {rejected_requests})")

    def plot_strategy_grid(self):
        """
        Plot the strategy grid, utilities, and arrows for most frequent cooperation partners.
        Arrow intensity corresponds to the cooperation frequency percentage.
        """
        if self.strategy_partners is None:
            raise ValueError("You must run analyze_strategy_performance() before plotting.")

        # Prepare the grid
        L = self.game.grid.L
        players = self.game.grid.players
        strategy_grid = np.array([player.strategy_name for player in players]).reshape(L, L)
        utility_grid = np.array([player.total_utility for player in players]).reshape(L, L)

        # Plot the grid
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(utility_grid, cmap="coolwarm", alpha=0.5)
        ax.set_title("Strategy Grid with Utilities and Cooperation Arrows", pad=20)

        # Add strategy names and utilities
        for i in range(L):
            for j in range(L):
                player = players[i * L + j]
                strategy = player.strategy_name
                utility = player.total_utility
                ax.text(j, i, f"{strategy}\n{utility:.1f}", ha="center", va="center", fontsize=8)

        # Arrow properties
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        for i in range(L):
            for j in range(L):
                player = players[i * L + j]
                if player.id in self.strategy_partners and self.strategy_partners[player.id]:
                    most_cooperative = max(self.strategy_partners[player.id], key=self.strategy_partners[player.id].get)
                    partner_count = self.strategy_partners[player.id][most_cooperative]
                    total_interactions = sum(self.strategy_partners[player.id].values())
                    intensity = partner_count / total_interactions  # Cooperation percentage

                    partner_pos = divmod(most_cooperative, L)
                    ax.arrow(
                        j, i,
                        partner_pos[1] - j, partner_pos[0] - i,
                        head_width=0.1, head_length=0.1,
                        fc=cmap(norm(intensity)), ec=cmap(norm(intensity)),
                        alpha=0.8, length_includes_head=True
                    )

        # Add colorbar for arrow intensity
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cooperation Intensity (0-1)", rotation=270, labelpad=20)

        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.xaxis.set_ticks_position('bottom')
        plt.show()

class UtilityFunction:
    def calculate(self, action, favor_size):
        """
        Calculate the utility change for the given action.
        :param action: The action taken ("cooperate", "reject", etc.).
        :param favor_size: The size of the favor involved.
        :return: Tuple (utility_for_requester, utility_for_responder)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class GameGrid:
    def __init__(self, L, N, diagonal_neighbors=True):
        """
        Initialize the game grid with players.
        :param L: Size of the grid (LxL)
        :param N: Neighborhood radius
        :param diagonal_neighbors: Whether diagonals are considered as neighbors.
        """
        self.L = L
        self.N = N
        self.diagonal_neighbors = diagonal_neighbors
        self.players = [
            Player(player_id, self.strategy_generator())
            for player_id in range(L * L)
        ]

    def strategy_generator(self):
        """
        Dynamically load all strategies from the strategies.py file and select one at random.
        """
        # Get all classes from the strategies module
        strategies_module = importlib.import_module("strategies")
        strategies = [
            cls
            for _, cls in inspect.getmembers(strategies_module, inspect.isclass)
            if issubclass(cls, Strategy) and cls is not Strategy
        ]

        # Choose a random strategy
        chosen_strategy = random.choice(strategies)
        return chosen_strategy()

    def __strategy_generator_old(self):
        """
        Generates a random strategy for each player.
        This method can be extended or modified for custom logic.
        """
        strategies = [RandomStrategy(), SelfishStrategy()]
        strategies = [ExceptAlwaysStrategy()]
        return random.choice(strategies)

    def shuffle_players(self):
        """Shuffle the order of players for a round."""
        return random.sample(self.players, len(self.players))

    def get_neighbors(self, player_id):
        """
        Find neighbors for a player based on the neighborhood radius and diagonal settings.
        :param player_id: ID of the player.
        :return: List of neighboring player IDs.
        """
        x, y = divmod(player_id, self.L)
        neighbors = []
        for i in range(-self.N, self.N + 1):
            for j in range(-self.N, self.N + 1):
                if not self.diagonal_neighbors:
                    # Manhattan distance: Diagonal neighbors count as > 1
                    if abs(i) + abs(j) > self.N:
                        continue
                else:
                    # Include diagonals: Max coordinate difference
                    if max(abs(i), abs(j)) > self.N:
                        continue

                # Periodic boundary conditions
                nx = (x + i) % self.L
                ny = (y + j) % self.L
                if (nx, ny) != (x, y):
                    neighbors.append(nx * self.L + ny)
        return neighbors

    def display_grid(self):
        """Displays the grid with player IDs."""
        grid = np.array([player.id for player in self.players]).reshape(self.L, self.L)
        print(grid)
    
class Player:
    def __init__(self, player_id, strategy):
        """
        Initialize a player with their ID, strategy, and initial utility.
        :param player_id: Unique ID of the player (from the grid).
        :param strategy: Strategy object or function assigned to this player.
        """
        self.id = player_id
        self.total_utility = 0  # Start with zero utility
        self.strategy = strategy  # Strategy decides all actions
        self.strategy_name = strategy.name  # Name of the strategy
        self.interaction_history = []  # Record of interactions [(other_player_id, outcome)]

    def decide_ask_for_help(self, neighbors):
        """
        Ask the strategy to decide whether to ask for help and from whom.
        :param neighbors: List of neighboring player IDs.
        :return: Dictionary with the decision:
            {
                "favor_size": int or None,
                "target": int or None,
                "action": str ("ask" or "none")
            }
        """
        return self.strategy.ask_for_help(self, neighbors, self.interaction_history)

    def decide_respond_to_help(self, requester_id, favor_size):
        """
        Ask the strategy to decide whether to respond positively to a help request.
        :param requester_id: ID of the player requesting help.
        :param favor_size: The size of the favor being requested.
        :return: Boolean (True if the player agrees to help, False otherwise).
        """
        return self.strategy.respond_to_help(self, requester_id, favor_size, self.interaction_history)

    def update_utility(self, utility_change):
        """
        Update the player's total utility.
        :param utility_change: Amount to add or subtract from the player's utility.
        """
        self.total_utility += utility_change

    def record_interaction(self, other_player_id, outcome):
        """
        Record the result of an interaction with another player.
        :param other_player_id: ID of the other player.
        :param outcome: Result of the interaction (e.g., "cooperate", "reject").
        """
        self.interaction_history.append((other_player_id, outcome))

