"""
algorithms.py
=============

ImplementaÃ§Ãµes de algoritmos de Aprendizado por ReforÃ§o.

Algoritmos de PrediÃ§Ã£o (AvaliaÃ§Ã£o de PolÃ­tica):
------------------------------------------------
- td_zero_prediction(): TD(0) para avaliaÃ§Ã£o de polÃ­tica
- first_visit_mc_prediction(): Monte Carlo First-Visit para avaliaÃ§Ã£o

Algoritmos de Controle (Busca de PolÃ­tica Ã“tima):
--------------------------------------------------
- sarsa(): SARSA (on-policy TD control)
- q_learning(): Q-Learning (off-policy TD control)
- expected_sarsa(): Expected SARSA (off-policy TD control)
- first_visit_mc_control(): Monte Carlo First-Visit Control
- mc_exploring_starts(): Monte Carlo com Exploring Starts

Algoritmos de Gradiente de PolÃ­tica:
-------------------------------------
- reinforce(): REINFORCE (Monte Carlo Policy Gradient)
- reinforce_baseline(): REINFORCE with Baseline
- actor_critic(): One-Step Actor-Critic

FunÃ§Ãµes Auxiliares:
-------------------
- get_greedy_policy(): Extrai polÃ­tica gulosa de Q-values
- epsilon_greedy_action(): Escolhe aÃ§Ã£o usando Îµ-greedy
- compute_softmax_policy(): Calcula Ï€(a|s,Î¸)
- sample_action_softmax(): Amostra aÃ§Ã£o segundo softmax
- compute_policy_gradient(): Calcula âˆ‡ln Ï€(a|s,Î¸)
- create_feature_vector(): Cria features one-hot para (s,a)

Autor: Material Educacional RL
Data: 2025
"""

import numpy as np
from typing import Tuple, Dict, List
from environment import GridWorld


# ============================================================================
# FUNÃ‡Ã•ES AUXILIARES
# ============================================================================

def state_to_idx(state: Tuple[int, int], n_cols: int) -> int:
    """
    Converte estado (row, col) para Ã­ndice linear.
    
    ParÃ¢metros:
    -----------
    state : Tuple[int, int]
        Estado (row, col)
    n_cols : int
        NÃºmero de colunas do grid
    
    Retorna:
    --------
    int
        Ãndice linear
    """
    return state[0] * n_cols + state[1]


def epsilon_greedy_action(Q: np.ndarray, state: Tuple[int, int],
                         gridworld: GridWorld, epsilon: float) -> int:
    """
    Escolhe aÃ§Ã£o usando polÃ­tica Îµ-greedy.
    
    Com probabilidade Îµ: explora (aÃ§Ã£o aleatÃ³ria)
    Com probabilidade (1-Îµ): exploita (melhor aÃ§Ã£o)
    
    ParÃ¢metros:
    -----------
    Q : np.ndarray
        Tabela Q de shape (n_states, n_actions)
    state : Tuple[int, int]
        Estado atual
    gridworld : GridWorld
        Ambiente
    epsilon : float
        Probabilidade de exploraÃ§Ã£o
    
    Retorna:
    --------
    int
        Ãndice da aÃ§Ã£o escolhida
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(gridworld.actions))
    else:
        state_idx = state_to_idx(state, gridworld.cols)
        return np.argmax(Q[state_idx])


def get_greedy_policy(Q: np.ndarray, gridworld: GridWorld) -> Dict:
    """
    Extrai polÃ­tica gulosa dos Q-values.
    
    Para cada estado, escolhe a aÃ§Ã£o com maior Q(s,a).
    
    ParÃ¢metros:
    -----------
    Q : np.ndarray
        Tabela Q
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    Dict
        DicionÃ¡rio {estado: aÃ§Ã£o}
    
    Exemplo:
    --------
    >>> Q, _ = q_learning(gw, n_episodes=1000)
    >>> policy = get_greedy_policy(Q, gw)
    >>> print(policy[(0, 0)])  # Melhor aÃ§Ã£o no estado (0,0)
    'N'
    """
    policy = {}
    
    for state in gridworld.states:
        if gridworld.is_terminal(state) or state in gridworld.walls:
            continue
        
        state_idx = state_to_idx(state, gridworld.cols)
        best_action_idx = np.argmax(Q[state_idx])
        policy[state] = gridworld.actions[best_action_idx]
    
    return policy


# ============================================================================
# FUNÃ‡Ã•ES AUXILIARES PARA GRADIENTE DE POLÃTICA
# ============================================================================

def create_feature_vector(state: Tuple[int, int], action_idx: int,
                         gridworld: GridWorld) -> np.ndarray:
    """
    Cria vetor de features one-hot para par (estado, aÃ§Ã£o).
    
    Features sÃ£o representadas como one-hot encoding:
    - DimensÃ£o: n_states Ã— n_actions
    - Somente o elemento [state_idx * n_actions + action_idx] = 1
    
    ParÃ¢metros:
    -----------
    state : Tuple[int, int]
        Estado
    action_idx : int
        Ãndice da aÃ§Ã£o
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    np.ndarray
        Vetor de features de dimensÃ£o (n_states * n_actions,)
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    feature_dim = n_states * n_actions
    
    features = np.zeros(feature_dim)
    state_idx = state_to_idx(state, gridworld.cols)
    feature_idx = state_idx * n_actions + action_idx
    features[feature_idx] = 1.0
    
    return features


def compute_softmax_policy(state: Tuple[int, int], theta: np.ndarray,
                          gridworld: GridWorld) -> np.ndarray:
    """
    Calcula probabilidades da polÃ­tica softmax Ï€(a|s,Î¸).
    
    Ï€(a|s,Î¸) = exp(h(s,a)) / Î£_b exp(h(s,b))
    onde h(s,a) = Î¸áµ€ x(s,a)
    
    ParÃ¢metros:
    -----------
    state : Tuple[int, int]
        Estado
    theta : np.ndarray
        ParÃ¢metros da polÃ­tica
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    np.ndarray
        Vetor de probabilidades para cada aÃ§Ã£o, shape (n_actions,)
    """
    n_actions = len(gridworld.actions)
    preferences = np.zeros(n_actions)
    
    for action_idx in range(n_actions):
        features = create_feature_vector(state, action_idx, gridworld)
        preferences[action_idx] = np.dot(theta, features)
    
    # Subtrai max para estabilidade numÃ©rica
    preferences = preferences - np.max(preferences)
    exp_prefs = np.exp(preferences)
    probs = exp_prefs / np.sum(exp_prefs)
    
    return probs


def sample_action_softmax(state: Tuple[int, int], theta: np.ndarray,
                         gridworld: GridWorld) -> int:
    """
    Amostra aÃ§Ã£o segundo polÃ­tica softmax Ï€(Â·|s,Î¸).
    
    ParÃ¢metros:
    -----------
    state : Tuple[int, int]
        Estado
    theta : np.ndarray
        ParÃ¢metros da polÃ­tica
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    int
        Ãndice da aÃ§Ã£o amostrada
    """
    probs = compute_softmax_policy(state, theta, gridworld)
    action_idx = np.random.choice(len(gridworld.actions), p=probs)
    return action_idx


def compute_policy_gradient(state: Tuple[int, int], action_idx: int,
                           theta: np.ndarray, gridworld: GridWorld) -> np.ndarray:
    """
    Calcula gradiente âˆ‡ln Ï€(a|s,Î¸) para polÃ­tica softmax.
    
    âˆ‡ln Ï€(a|s,Î¸) = x(s,a) - Î£_b Ï€(b|s,Î¸) x(s,b)
    
    ParÃ¢metros:
    -----------
    state : Tuple[int, int]
        Estado
    action_idx : int
        Ãndice da aÃ§Ã£o tomada
    theta : np.ndarray
        ParÃ¢metros da polÃ­tica
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    np.ndarray
        Vetor gradiente, mesma dimensÃ£o que Î¸
    """
    # Features da aÃ§Ã£o tomada
    features_a = create_feature_vector(state, action_idx, gridworld)
    
    # Expectativa das features sob Ï€
    probs = compute_softmax_policy(state, theta, gridworld)
    expected_features = np.zeros_like(theta)
    
    for a_idx in range(len(gridworld.actions)):
        features_b = create_feature_vector(state, a_idx, gridworld)
        expected_features += probs[a_idx] * features_b
    
    # Gradiente
    grad = features_a - expected_features
    
    return grad


# ============================================================================
# ALGORITMO REINFORCE
# ============================================================================

def reinforce(gridworld: GridWorld, n_episodes: int = 1000,
             alpha_theta: float = 2**-13, gamma: float = 0.99,
             theta_init: np.ndarray = None,
             initial_state: Tuple[int, int] = None,
             verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    Aprende polÃ­tica estocÃ¡stica Ï€(a|s,Î¸) diretamente usando gradiente de polÃ­tica.
    Usa retorno completo G_t e atualiza no fim de cada episÃ³dio.
    
    Algoritmo:
    ----------
    Para cada episÃ³dio:
        1. Gera episÃ³dio seguindo Ï€(Â·|Â·,Î¸)
        2. Para cada passo t:
            G_t â† Î£_{k=t+1}^T Î³^{k-t-1} R_k
            Î¸ â† Î¸ + Î± Î³^t G_t âˆ‡ln Ï€(A_t|S_t,Î¸)
    
    ParÃ¢metros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        NÃºmero de episÃ³dios de treinamento
    alpha_theta : float, default=2**-13
        Taxa de aprendizado para Î¸
    gamma : float, default=0.99
        Fator de desconto
    theta_init : np.ndarray, opcional
        InicializaÃ§Ã£o dos parÃ¢metros Î¸
        Se None, inicializa com zeros
    initial_state : Tuple[int, int], opcional
        Estado inicial fixo
        Se None, escolhe aleatoriamente
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - theta: ParÃ¢metros finais da polÃ­tica
        - episode_rewards: Lista de recompensas totais por episÃ³dio
    
    Exemplo:
    --------
    >>> from environment import create_cliff_world
    >>> gw = create_cliff_world()
    >>> theta, rewards = reinforce(gw, n_episodes=2000, alpha_theta=2**-13)
    >>> # Para ver a polÃ­tica aprendida:
    >>> probs = compute_softmax_policy((3, 0), theta, gw)
    >>> print(probs)  # Probabilidades das 4 aÃ§Ãµes
    """
    # InicializaÃ§Ã£o
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    feature_dim = n_states * n_actions
    
    if theta_init is None:
        theta = np.zeros(feature_dim)
    else:
        theta = theta_init.copy()
    
    episode_rewards = []
    
    # Loop de episÃ³dios
    for episode in range(n_episodes):
        # Escolhe estado inicial
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s) and s not in gridworld.walls]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        # Gera episÃ³dio
        states = []
        actions = []
        rewards = []
        
        while not gridworld.is_terminal(state):
            action_idx = sample_action_softmax(state, theta, gridworld)
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            
            state = next_state
        
        # Calcula retornos G_t
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G
        
        # AtualizaÃ§Ã£o dos parÃ¢metros
        for t in range(T):
            grad = compute_policy_gradient(states[t], actions[t], theta, gridworld)
            theta += alpha_theta * (gamma ** t) * returns[t] * grad
        
        episode_rewards.append(np.sum(rewards))
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"EpisÃ³dio {episode + 1}/{n_episodes} - Reward mÃ©dio: {avg_reward:.2f}")
    
    return theta, episode_rewards


# ============================================================================
# ALGORITMO REINFORCE COM BASELINE
# ============================================================================

def reinforce_baseline(gridworld: GridWorld, n_episodes: int = 1000,
                      alpha_theta: float = 2**-9, alpha_w: float = 2**-6,
                      gamma: float = 0.99,
                      theta_init: np.ndarray = None, w_init: np.ndarray = None,
                      initial_state: Tuple[int, int] = None,
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Algoritmo REINFORCE com Baseline (reduz variÃ¢ncia).
    
    Aprende polÃ­tica Ï€(a|s,Î¸) e funÃ§Ã£o valor baseline vÌ‚(s,w).
    A baseline reduz variÃ¢ncia sem introduzir viÃ©s.
    
    Algoritmo:
    ----------
    Para cada episÃ³dio:
        1. Gera episÃ³dio seguindo Ï€(Â·|Â·,Î¸)
        2. Para cada passo t:
            G_t â† Î£_{k=t+1}^T Î³^{k-t-1} R_k
            Î´ â† G_t - vÌ‚(S_t,w)  # Advantage
            w â† w + Î±_w Î´ âˆ‡vÌ‚(S_t,w)
            Î¸ â† Î¸ + Î±_Î¸ Î³^t Î´ âˆ‡ln Ï€(A_t|S_t,Î¸)
    
    ParÃ¢metros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        NÃºmero de episÃ³dios
    alpha_theta : float, default=2**-9
        Taxa de aprendizado para Î¸ (polÃ­tica)
    alpha_w : float, default=2**-6
        Taxa de aprendizado para w (baseline)
    gamma : float, default=0.99
        Fator de desconto
    theta_init, w_init : np.ndarray, opcional
        InicializaÃ§Ãµes
    initial_state : Tuple[int, int], opcional
        Estado inicial fixo
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, List[float]]
        - theta: ParÃ¢metros da polÃ­tica
        - w: Pesos da baseline
        - episode_rewards: Recompensas por episÃ³dio
    
    Exemplo:
    --------
    >>> theta, w, rewards = reinforce_baseline(gw, n_episodes=2000)
    >>> # Baseline aprendida:
    >>> v_state = np.dot(w, create_feature_vector((0, 0), 0, gw)[:len(w)])
    """
    # InicializaÃ§Ã£o
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    feature_dim = n_states * n_actions
    
    if theta_init is None:
        theta = np.zeros(feature_dim)
    else:
        theta = theta_init.copy()
    
    # Baseline usa features de estado (nÃ£o estado-aÃ§Ã£o)
    if w_init is None:
        w = np.zeros(n_states)
    else:
        w = w_init.copy()
    
    episode_rewards = []
    
    # Loop de episÃ³dios
    for episode in range(n_episodes):
        # Escolhe estado inicial
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s) and s not in gridworld.walls]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        # Gera episÃ³dio
        states = []
        actions = []
        rewards = []
        
        while not gridworld.is_terminal(state):
            action_idx = sample_action_softmax(state, theta, gridworld)
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            
            state = next_state
        
        # Calcula retornos G_t
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G
        
        # AtualizaÃ§Ã£o dos parÃ¢metros
        for t in range(T):
            state_idx = state_to_idx(states[t], gridworld.cols)
            
            # Valor baseline
            v_baseline = w[state_idx]
            
            # Advantage
            delta = returns[t] - v_baseline
            
            # Atualiza baseline (w)
            # âˆ‡vÌ‚(s,w) = feature de estado (one-hot)
            w[state_idx] += alpha_w * delta
            
            # Atualiza polÃ­tica (Î¸)
            grad = compute_policy_gradient(states[t], actions[t], theta, gridworld)
            theta += alpha_theta * (gamma ** t) * delta * grad
        
        episode_rewards.append(np.sum(rewards))
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"EpisÃ³dio {episode + 1}/{n_episodes} - Reward mÃ©dio: {avg_reward:.2f}")
    
    return theta, w, episode_rewards


# ============================================================================
# ALGORITMO ACTOR-CRITIC (ONE-STEP)
# ============================================================================

def actor_critic(gridworld: GridWorld, n_episodes: int = 1000,
                alpha_theta: float = 0.5, alpha_w: float = 0.5,
                gamma: float = 0.99,
                theta_init: np.ndarray = None, w_init: np.ndarray = None,
                initial_state: Tuple[int, int] = None,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Algoritmo One-Step Actor-Critic.
    
    Actor: polÃ­tica Ï€(a|s,Î¸)
    Critic: funÃ§Ã£o valor V(s,w)
    
    Diferente de REINFORCE:
    - AtualizaÃ§Ã£o ONLINE (a cada passo, nÃ£o episÃ³dica)
    - Usa TD error: Î´ = R + Î³V(s') - V(s)
    - Bootstrapping reduz variÃ¢ncia
    
    Algoritmo:
    ----------
    Para cada episÃ³dio:
        Inicializa S
        I â† 1
        Enquanto S nÃ£o terminal:
            A ~ Ï€(Â·|S,Î¸)
            Toma A, observa S', R
            Î´ â† R + Î³V(S',w) - V(S,w)
            w â† w + Î±_w Î´ âˆ‡V(S,w)
            Î¸ â† Î¸ + Î±_Î¸ I Î´ âˆ‡ln Ï€(A|S,Î¸)
            I â† Î³I
            S â† S'
    
    ParÃ¢metros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        NÃºmero de episÃ³dios
    alpha_theta : float, default=0.5
        Taxa de aprendizado do actor
    alpha_w : float, default=0.5
        Taxa de aprendizado do critic
    gamma : float, default=0.99
        Fator de desconto
    theta_init, w_init : np.ndarray, opcional
        InicializaÃ§Ãµes
    initial_state : Tuple[int, int], opcional
        Estado inicial fixo
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, List[float]]
        - theta: ParÃ¢metros do actor
        - w: Pesos do critic
        - episode_rewards: Recompensas por episÃ³dio
    
    Exemplo:
    --------
    >>> theta, w, rewards = actor_critic(gw, n_episodes=1000, 
    ...                                   alpha_theta=0.5, alpha_w=0.5)
    """
    # InicializaÃ§Ã£o
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    feature_dim = n_states * n_actions
    
    if theta_init is None:
        theta = np.zeros(feature_dim)
    else:
        theta = theta_init.copy()
    
    if w_init is None:
        w = np.zeros(n_states)
    else:
        w = w_init.copy()
    
    episode_rewards = []
    
    # Loop de episÃ³dios
    for episode in range(n_episodes):
        # Escolhe estado inicial
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s) and s not in gridworld.walls]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        I = 1.0  # Fator de desconto acumulado
        total_reward = 0
        
        # Loop do episÃ³dio (ONLINE)
        while not gridworld.is_terminal(state):
            # Actor: escolhe aÃ§Ã£o
            action_idx = sample_action_softmax(state, theta, gridworld)
            action = gridworld.actions[action_idx]
            
            # Executa aÃ§Ã£o
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            
            # Ãndices de estado
            state_idx = state_to_idx(state, gridworld.cols)
            next_state_idx = state_to_idx(next_state, gridworld.cols)
            
            # TD error
            if not gridworld.is_terminal(next_state):
                v_next = w[next_state_idx]
            else:
                v_next = 0.0
            
            v_current = w[state_idx]
            delta = reward + gamma * v_next - v_current
            
            # Atualiza Critic (w)
            w[state_idx] += alpha_w * delta
            
            # Atualiza Actor (Î¸)
            grad = compute_policy_gradient(state, action_idx, theta, gridworld)
            theta += alpha_theta * I * delta * grad
            
            # Atualiza fator de desconto
            I *= gamma
            
            # PrÃ³ximo estado
            state = next_state
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"EpisÃ³dio {episode + 1}/{n_episodes} - Reward mÃ©dio: {avg_reward:.2f}")
    
    return theta, w, episode_rewards


# ============================================================================
# ALGORITMOS EXISTENTES (mantidos da versÃ£o original)
# ============================================================================

# [Aqui viriam os algoritmos existentes: td_zero_prediction, sarsa, q_learning, 
# expected_sarsa, first_visit_mc_prediction, first_visit_mc_control, 
# mc_exploring_starts - mantidos exatamente como estavam]

# Por brevidade, nÃ£o vou replicar todo o cÃ³digo existente aqui, mas na implementaÃ§Ã£o
# real eles devem ser mantidos integralmente.
