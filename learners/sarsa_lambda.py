from common.types import State
from typing import Any, Dict, List, Tuple

import numpy as np
from basis_functions.basis_set import BasisSet


class SarsaLambda:
    """
        A general class that implements Sarsa(lambda) with linear function approximation.
    """

    def __init__(self, basissets: List[BasisSet], number_of_actions: int, 
                    alpha: float = 0.015, gamma: float = 0.99, lambd: float = 0.85, 
                    epsilon: float = 0.05, perform_alpha_scaling: bool = False, 
                    should_init_randomly: bool = True, deterministic_best_action: bool = False,
                    num_adaptive_episodes: int = -1,
                    ) -> None:
        """Initialises this Sarsa(Lamba) learner object.

        Args:
            basissets (List[BasisSet]): The basis functions to use. Must have the same length as number_of_actions
            number_of_actions (int): The number of distinct discrete actions we have.
            alpha (float, optional): Learning rate. Defaults to 0.015.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lambd (float, optional): Lambda, eligibility traces decay. Defaults to 0.85.
            epsilon (float, optional): What fraction of steps do we take a random instead of a greedy action. Defaults to 0.05.
            perform_alpha_scaling (bool, optional): If true, does PARL2 Alpha Scaling from Dabney. Defaults to False.
            should_init_randomly (bool, optional): If true, initialises weights randomly instead of to 0. Defaults to True.
            deterministic_best_action (bool, optional): If true and multiple actions have the same value, choose the first one. If it is false, choose randomly from the best actions. Defaults to False.
        """
        self.basissets = basissets
        self.num_actions = number_of_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.do_adaptive_alpha = perform_alpha_scaling
        self.deterministic_best_action = deterministic_best_action
        self.should_init_randomly = should_init_randomly
        self.num_adaptive_episodes = num_adaptive_episodes
        self.weights: List[np.ndarray] = [
            np.random.randn(bset.num_features()) if should_init_randomly else np.zeros(bset.num_features()) for bset in self.basissets
        ]

        self.eligibility_traces: List[np.ndarray] = [
            np.zeros(bset.num_features()) for bset in self.basissets
        ]
        self.episode_count = 0
    
    def get_action(self, state: State) -> int:
        """Returns an action corresponding to the best one.

        Args:
            state (State): The state

        Returns:
            int: The action, in [0, self.num_actions) 
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        features = [
            bset.get_value(state) for bset in self.basissets
        ]
        return self.V(features)[1]

    def add_sample(self, s: State, a: int, reward: float, s_next: State, a_next: int, is_terminal: bool) -> None:
        """Adds a sample and updates the weights and elig. traces.

        Args:
            s (State): The state
            a (int): Action taken in state s
            reward (float): The reward gotten from action a in state s.
            s_next (State): The next state
            a_next (int): The action that will be performed in the next state.
            is_terminal (bool): Is the current state terminal, i.e. is the episode done
        """
        # Get the features
        features_now = self.basissets[a].get_value(s)
        features_next = self.basissets[a_next].get_value(s_next)
        Q_now = self.Q(features_now, a)

        delta = reward - Q_now
        # Only add next Q if not terminal
        if not is_terminal:
            delta += self.gamma * self.Q(features_next, a_next)
        
        # Decay eligibility traces
        for temp_action in range(self.num_actions):
            self.eligibility_traces[temp_action] *= self.lambd * self.gamma
            if temp_action == a:
                # add features (phi) to z
                self.eligibility_traces[temp_action] += features_now
            
        # Update weights
        for temp_action in range(self.num_actions):
            self.weights[temp_action] += self.alpha * delta * self.eligibility_traces[temp_action] * \
                                            self.basissets[temp_action].get_shrink()
        
        self.alpha = self.alpha if not self.do_adaptive_alpha else self.get_new_adaptive_alpha(is_terminal,
            features_now, features_next,
            a, a_next
        )
        
        if self.episode_count <= self.num_adaptive_episodes:
            num_funcs, new_weights, new_elig = self.basissets[a].discover(s, is_terminal, delta, features_now, 
                                                                            self.weights[a], self.eligibility_traces[a])
            if num_funcs != 0:
                print(f"At episode {self.episode_count}, added = {num_funcs}")
            self.weights[a] = new_weights
            self.eligibility_traces[a] = new_elig

    

    def zero_trace(self) -> None:
        """
            Makes the traces 0, needed when the episode ends
        """
        self.episode_count += 1
        for a in range(self.num_actions):
            self.eligibility_traces[a] *= 0
    

    def Q(self, features: np.ndarray, action: int) -> float:
        """The Q value of this state and action

        Args:
            features (np.ndarray): Features given by the state S.
            action (int): The action to check the Q value for

        Returns:
            float: Q(s, a)
        """
        return self.weights[action].dot(features)

    def V(self, features: List[np.ndarray]) -> Tuple[float, int]:
        """Returns the best Q value, and corresponding action for this state.

        Args:
            features (List[np.ndarray]): 

        Returns:
            Tuple[float, int]: best Q, best action
        """
        Qs = np.array([self.Q(features[a], a) for a in range(self.num_actions)])
        best_actions = np.where(np.isclose(Qs, Qs.max()))[0]
        best_action = -100
        if self.deterministic_best_action or len(best_actions) == 1:
            best_action = best_actions[0]
        else:
            if len(best_actions) == 0:
                print(f"Invalid values recorded, Qs = {Qs}")
                best_action = 0
            else:    
                best_action = np.random.choice(best_actions)
        
        return Qs[best_action], best_action

    def get_new_adaptive_alpha(self, is_terminal: bool, features_now, features_next, action_now, action_next) -> float:
        """Updates the alpha using alpha scaling

        Args:
            is_terminal (bool): Is state s terminal
            features_now ([type]): phi(s)
            features_next ([type]): phi(s')
            action_now ([type]): a
            action_next ([type]): a'

        Returns:
            float: The returned alpha. This function also updates self.alpha
        """
        delta_phi = -self.eligibility_traces[action_now].dot(features_now)

        if not is_terminal:
            delta_phi += self.gamma * self.eligibility_traces[action_next].dot(features_next)
        
        if delta_phi < 0:
            self.alpha = min(self.alpha, -1/delta_phi)

        return self.alpha
    
    def __repr__(self) -> str:
        return f"SarsaLambda(basissets={self.basissets}, number_of_actions={self.num_actions}, alpha={self.alpha}, gamma={self.gamma}, lambd={self.lambd}, epsilon={self.epsilon}, perform_alpha_scaling={self.do_adaptive_alpha},  should_init_randomly={self.should_init_randomly}, deterministic_best_action={self.deterministic_best_action})"
    
    def params(self) -> Dict[str, Any]:
        return dict(
            num_actions = self.num_actions,
            alpha = self.alpha,
            gamma = self.gamma,
            lambd = self.lambd,
            epsilon = self.epsilon,
            do_adaptive_alpha = self.do_adaptive_alpha,
            deterministic_best_action = self.deterministic_best_action,
            should_init_randomly = self.should_init_randomly,
            num_adaptive_episodes = self.num_adaptive_episodes,
        )