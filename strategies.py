import random

class Strategy:
    def __init__(self, name):
        self.name = name

    def ask_for_help(self, player, neighbors, interaction_history):
        """
        Decide whether to ask for help, from whom, and how much.
        :param player: The player making the decision.
        :param neighbors: List of neighboring player IDs.
        :param interaction_history: List of past interactions.
        :return: Dictionary with decision:
            {
                "favor_size": int or None,
                "target": int or None,
                "action": str ("ask" or "none")
            }
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def respond_to_help(self, player, requester_id, favor_size, interaction_history):
        """
        Decide whether to respond positively to a help request.
        :param player: The player making the decision.
        :param requester_id: ID of the player requesting help.
        :param favor_size: The size of the favor being requested.
        :param interaction_history: List of past interactions.
        :return: Boolean (True to agree, False to reject).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class SelfishStrategy(Strategy):
    def __init__(self):
        super().__init__("Selfish")

    def ask_for_help(self, player, neighbors, interaction_history):
        if not neighbors:
            return {"favor_size": None, "target": None, "action": "none"}
        # Ask only if player needs a big favor
        favor_size = random.randint(2, 3)  # Only large favors
        target = random.choice(neighbors)
        return {"favor_size": favor_size, "target": target, "action": "ask"}

    def respond_to_help(self, player, requester_id, favor_size, interaction_history):
        return False
    
class RandomStrategy(Strategy):
    def __init__(self):
        super().__init__("Random")

    def ask_for_help(self, player, neighbors, interaction_history):
        if not neighbors or random.random() < 0.5:  # 50% chance to skip asking
            return {"favor_size": None, "target": None, "action": "none"}
        target = random.choice(neighbors)
        favor_size = random.randint(1, 3)  # Random favor size
        return {"favor_size": favor_size, "target": target, "action": "ask"}

    def respond_to_help(self, player, requester_id, favor_size, interaction_history):
        # Always help
        return True
    
class ExceptAlwaysStrategy(Strategy):
    def __init__(self):
        super().__init__("ExceptAlways")

    def ask_for_help(self, player, neighbors, interaction_history):
        if not neighbors:
            return {"favor_size": None, "target": None, "action": "none"}
        # Always ask the first neighbor for a favor of size 1
        return {"favor_size": 1, "target": neighbors[0], "action": "ask"}

    def respond_to_help(self, player, requester_id, favor_size, interaction_history):
        return True

class TitForTatStrategy(Strategy):
    def __init__(self):
        super().__init__("Tit-for-Tat")

    def ask_for_help(self, player, neighbors, interaction_history):
        """
        Decide whom to ask for help based on the Tit-for-Tat principle.
        Always cooperates unless there's no clear history.
        """
        if not neighbors:
            return {"favor_size": None, "target": None, "action": "none"}

        # Ask the first neighbor in the list with a random favor size
        target = neighbors[0]
        favor_size = 1  # Fixed favor size for simplicity
        return {"favor_size": favor_size, "target": target, "action": "ask"}

    def respond_to_help(self, player, requester_id, favor_size, interaction_history):
        """
        Respond to help based on the Tit-for-Tat principle.
        Cooperates if the requester cooperated in the past, otherwise refuses.
        Ignores "busy" outcomes.
        """
        for past_requester_id, outcome in interaction_history:
            if past_requester_id == requester_id:
                # If requester cooperated before, cooperate
                if outcome == "cooperate":
                    return True
                # If requester rejected before, reject
                elif outcome == "reject":
                    return False

        # Default to cooperation if no history
        return True
