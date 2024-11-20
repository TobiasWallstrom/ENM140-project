import numpy as np
import random
from strategies import *

class Game:
    def __init__(self, grid, utility_function):
        """
        Initialize the game with a grid and a utility function.
        :param grid: An instance of GameGrid containing players and neighbors.
        :param utility_function: An instance of UtilityFunction to calculate utility changes.
        """
        self.grid = grid  # GameGrid instance
        self.utility_function = utility_function  # Utility function instance
        self.history = []  # List to store interaction history for each round

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

                    # Mark the target player as interacted
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

    def display_game_history(self):
        """
        Display the history of all interactions across all rounds.
        """
        for round_num, round_events in enumerate(self.history, start=1):
            print(f"\n--- Round {round_num} ---")
            for event in round_events:
                print(f"Player {event['initiator']} -> Player {event['target']} | "
                      f"Action: {event['action']} | Result: {event['result']} | "
                      f"Favor Size: {event['favor_size']}")

    def display_total_results(self):
        """
        Display the total utilities of all players.
        """
        print("\n--- Final Results ---")
        for player in self.grid.players:
            print(f"Player {player.id}: Total Utility = {player.total_utility}")

    def summarize_player(self, player_id):
        """
        Summarize a specific player's history and total utility.
        :param player_id: ID of the player to summarize.
        """
        # Find the player's total utility
        player = next(p for p in self.grid.players if p.id == player_id)
        total_utility = player.total_utility

        # Counters for how often the player was asked and outcomes
        times_asked = 0
        successful_requests = 0
        rejected_requests = 0

        # Build a detailed history of the player's actions across all rounds
        print(f"\n--- Summary for Player {player_id} ---")
        for round_num, round_events in enumerate(self.history, start=1):
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

                    # Check if the player was the target and count outcomes
                    if event["target"] == player_id:
                        times_asked += 1
                        if result == "cooperate":
                            successful_requests += 1
                        elif result == "reject":
                            rejected_requests += 1
                        # Show who asked this player
                        print(f"Round {round_num}: Asked by Player {initiator}, Action = {action}, "
                            f"Result = {result}, Favor Size = {favor_size}")
                    elif event["initiator"] == player_id:
                        # Print events where this player was the initiator
                        print(f"Round {round_num}: Action = {action}, Target = {target}, "
                            f"Result = {result}, Favor Size = {favor_size}")

        # Print the total utility and request stats
        print(f"Total Utility: {total_utility}")
        print(f"Times Asked for Help: {times_asked} (Cooperated: {successful_requests}, Rejected: {rejected_requests})")
 
    def display_utility_grid(self):
        """
        Display the grid with each player's total utility instead of their ID.
        """
        utilities = [player.total_utility for player in self.grid.players]
        utility_grid = np.array(utilities).reshape(self.grid.L, self.grid.L)
        print("\n--- Utility Grid ---")
        print(utility_grid)

class UtilityFunction:
    def calculate(self, action, favor_size):
        """
        Calculate the utility change for the given action.
        :param action: The action taken ("cooperate", "reject", etc.).
        :param favor_size: The size of the favor involved.
        :return: Tuple (utility_for_requester, utility_for_responder)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
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

