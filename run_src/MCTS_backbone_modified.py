"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Modified to use the number of trajectories leading to the same choice as the reward.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random

node_cnt = 0

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

class MCTS_Node(ABC):
    """
    A representation of a single state in the MCTS.
    This is an abstract base class; you should implement the abstract methods for your specific use case.
    """

    def __init__(self) -> None:
        super().__init__()

        global node_cnt
        self.id = node_cnt
        node_cnt += 1

        self.rollout_id = None

    def set_rollout_id(self, rollout_id: int):
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int):
        "All possible successors of this state"
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children (i.e., it's a terminal node)"
        raise NotImplementedError

    @abstractmethod
    def get_choice(self):
        "Returns the choice made at this node (only valid for terminal nodes)"
        raise NotImplementedError

    @abstractmethod
    def skip_backprop(self):
        "If True, the reward of this node will not be accumulated in the backpropagation step."
        raise NotImplementedError

class MCTS_Searcher:
    "Monte Carlo tree searcher with reward based on the number of trajectories leading to the same choice"

    def __init__(
        self,
        exploration_weight: float = 1.0,
        weight_scheduler: str = "const",
        num_rollouts: int = 1000,
        discount: float = 1.0,
        verbose: bool = False,
    ):
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)  # total reward of each node
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)  # total visit count for each node
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = dict()  # children of each node

        # Explored nodes are nodes that have been expanded and simulated
        self.explored_nodes = set()

        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount

        self.verbose = verbose

        global node_cnt
        node_cnt = 0

        # Dictionary to keep track of the number of times each choice has been reached
        self.choice_counts = defaultdict(int)

    def do_rollout(self, root_node: MCTS_Node, rollout_id: int):
        "Perform one rollout from the root node"
        verbose_print("==> Selecting a node...", self.verbose)
        path = self._select(root_node, rollout_id)
        leaf = path[-1]
        verbose_print(f"==> Expanding node {leaf.id}...", self.verbose)
        self._expand(leaf, rollout_id)
        verbose_print(f"==> Simulating from node {leaf.id}...", self.verbose)
        simulate_path = self._simulate(leaf, rollout_id)
        verbose_print(f"==> Backpropagating ..", self.verbose)
        self._backpropagate(path + simulate_path)
        return simulate_path[-1] if simulate_path else leaf

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Find an unexplored descendant of `node`"
        path = []
        while True:
            path.append(node)
            # Case 1: Node has not been expanded yet
            if node not in self.parent2children.keys():
                return path

            # Case 2: Node has unexplored children
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                return path

            # Case 3: All children have been explored; select the best child
            node = self._uct_select(node, rollout_id)

    def _expand(self, node: MCTS_Node, rollout_id: int):
        "Expand the node by adding its children"
        if node in self.explored_nodes:
            return  # Already expanded

        if node.is_terminal():
            self.explored_nodes.add(node)
            return  # Terminal node; nothing to expand

        self.parent2children[node] = node.find_children(rollout_id)

    def _simulate(self, node: MCTS_Node, rollout_id: int) -> (float, List[MCTS_Node]):
        "Simulate a random path from the node to a terminal state and calculate the reward"
        path = []
        cur_node = node
        while True:
            if cur_node.is_terminal():
                self.explored_nodes.add(cur_node)
                # Get the choice made at the terminal node
                choice = cur_node.get_choice()
                if choice is not None:
                    # Update the count for this choice (including the current trajectory)
                    self.choice_counts[choice] += 1
                return path

            if cur_node not in self.parent2children.keys():
                self.parent2children[cur_node] = cur_node.find_children(rollout_id)

            cur_node = random.choice(self.parent2children[cur_node])  # Randomly select a child
            path.append(cur_node)

    def _backpropagate(self, path: List[MCTS_Node]):
        leaf = path[-1]
        choice = leaf.get_choice()
        if choice is not  None:
            reward = self.choice_counts[choice]
        else:
            reward = 0
        "Backpropagate the reward up the path"
        for node in reversed(path):
            self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)

    def _get_weight(self, rollout_id: int):
        # Adjust the exploration weight based on the scheduler
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "const":
            return self.exploration_weight

    def _uct_select(self, node: MCTS_Node, rollout_id: int):
        "Select a child of the node, balancing exploration & exploitation"
        # All children of the node should already be expanded
        assert all(n in self.explored_nodes for n in self.parent2children[node])

        return max(
            self.parent2children[node],
            key=lambda n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
        )

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int):
        "Compute the Upper Confidence Bound for Trees (UCT) value"
        if self.N[node] == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        else:
            weight = self._get_weight(rollout_id)
            return (self.Q[node] / self.N[node]) + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])
