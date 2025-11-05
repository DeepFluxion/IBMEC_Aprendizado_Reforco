"""
visualization_extended.py
=========================

Funções de visualização adicionais para algoritmos de Gradiente de Política.

Novas Funções para Policy Gradient:
------------------------------------
- visualize_stochastic_policy(): Visualiza probabilidades π(a|s) 
- plot_policy_arrows(): Mostra política com setas proporcionais
- plot_policy_gradient_learning(): Curvas de aprendizado para PG
- compare_variance(): Compara variância entre algoritmos

Autor: Material Educacional RL  
Data: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
from typing import Dict, List, Tuple
from environment import GridWorld


# ============================================================================
# VISUALIZAÇÃO DE POLÍTICA ESTOCÁSTICA
# ============================================================================

def visualize_stochastic_policy(theta: np.ndarray, gridworld: GridWorld,
                                title: str = "Política Estocástica π(a|s)",
                                figsize: Tuple[int, int] = None,
                                show_probs: bool = True):
    """
    Visualiza política estocástica mostrando probabilidades de cada ação.
    
    Para cada estado, mostra π(a|s) como:
    - Setas com tamanhos proporcionais às probabilidades
    - Valores numéricos (opcional)
    - Cores indicando magnitude
    
    Parâmetros:
    -----------
    theta : np.ndarray
        Parâmetros da política softmax
    gridworld : GridWorld
        Ambiente
    title : str
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    show_probs : bool, default=True
        Se True, mostra valores numéricos das probabilidades
    
    Exemplo:
    --------
    >>> from algorithms_extended import reinforce
    >>> theta, _ = reinforce(gw, n_episodes=2000)
    >>> visualize_stochastic_policy(theta, gw)
    """
    from algorithms_extended import compute_softmax_policy
    
    if figsize is None:
        figsize = (gridworld.cols * 3, gridworld.rows * 3)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_xlim(-0.5, gridworld.cols - 0.5)
    ax.set_ylim(gridworld.rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(gridworld.cols))
    ax.set_yticks(range(gridworld.rows))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Mapeamento de ações para direções
    action_vectors = {
        'N': (0, -0.3),   # Norte: cima
        'S': (0, 0.3),    # Sul: baixo
        'L': (0.3, 0),    # Leste: direita
        'O': (-0.3, 0)    # Oeste: esquerda
    }
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            
            # Cor de fundo
            if state in gridworld.walls:
                color = 'gray'
                rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                     boxstyle="round,pad=0.05", 
                                     facecolor=color, edgecolor='black',
                                     alpha=0.8, linewidth=2)
                ax.add_patch(rect)
                ax.text(col, row, '■', ha='center', va='center',
                       fontsize=20, fontweight='bold')
                continue
            
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black',
                                     alpha=0.6, linewidth=2)
                ax.add_patch(rect)
                ax.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                       fontsize=14, fontweight='bold')
                continue
            
            # Estados normais: mostra política estocástica
            else:
                rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor='white', edgecolor='black',
                                     alpha=0.3, linewidth=1)
                ax.add_patch(rect)
            
            # Calcula probabilidades
            probs = compute_softmax_policy(state, theta, gridworld)
            
            # Desenha setas proporcionais às probabilidades
            for action_idx, action in enumerate(gridworld.actions):
                prob = probs[action_idx]
                dx, dy = action_vectors[action]
                
                # Escala da seta proporcional à probabilidade
                scale = prob
                
                # Cor baseada na probabilidade
                if prob > 0.5:
                    color = 'darkgreen'
                    alpha = 0.9
                elif prob > 0.25:
                    color = 'green'
                    alpha = 0.7
                else:
                    color = 'gray'
                    alpha = 0.4
                
                # Desenha seta
                if prob > 0.01:  # Só desenha se probabilidade significativa
                    arrow = FancyArrow(col, row, dx * scale, dy * scale,
                                      width=0.02, head_width=0.12,
                                      head_length=0.08, fc=color, ec=color,
                                      alpha=alpha, linewidth=1.5)
                    ax.add_patch(arrow)
                
                # Mostra valor numérico
                if show_probs and prob > 0.05:
                    text_offset_x = dx * 0.5
                    text_offset_y = dy * 0.5
                    ax.text(col + text_offset_x, row + text_offset_y,
                           f'{prob:.2f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_policy_arrows(theta: np.ndarray, gridworld: GridWorld,
                      title: str = "Política (Ação Mais Provável)",
                      figsize: Tuple[int, int] = None):
    """
    Visualiza apenas a ação mais provável em cada estado (como política determinística).
    
    Similar a visualize_gridworld mas para política aprendida por gradiente.
    
    Parâmetros:
    -----------
    theta : np.ndarray
        Parâmetros da política
    gridworld : GridWorld
        Ambiente
    title : str
        Título
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    """
    from algorithms_extended import compute_softmax_policy
    
    if figsize is None:
        figsize = (gridworld.cols * 2, gridworld.rows * 2)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_xlim(-0.5, gridworld.cols - 0.5)
    ax.set_ylim(gridworld.rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(gridworld.cols))
    ax.set_yticks(range(gridworld.rows))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    action_symbols = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            
            if state in gridworld.walls:
                color = 'gray'
                rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, alpha=0.8)
                ax.add_patch(rect)
                continue
            
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, alpha=0.6)
                ax.add_patch(rect)
                ax.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                       fontsize=14, fontweight='bold')
                continue
            
            # Ação mais provável
            probs = compute_softmax_policy(state, theta, gridworld)
            best_action_idx = np.argmax(probs)
            best_action = gridworld.actions[best_action_idx]
            best_prob = probs[best_action_idx]
            
            # Cor baseada na confiança
            if best_prob > 0.7:
                color = 'darkgreen'
            elif best_prob > 0.4:
                color = 'green'
            else:
                color = 'orange'
            
            symbol = action_symbols[best_action]
            ax.text(col, row, symbol, ha='center', va='center',
                   fontsize=24, fontweight='bold', color=color)
            ax.text(col, row + 0.35, f'{best_prob:.2f}', ha='center', va='center',
                   fontsize=8, color='black')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# CURVAS DE APRENDIZADO PARA POLICY GRADIENT
# ============================================================================

def plot_policy_gradient_learning(rewards_dict: Dict[str, List[float]],
                                  window: int = 100,
                                  title: str = "Aprendizado - Policy Gradient",
                                  figsize: Tuple[int, int] = (14, 5)):
    """
    Plota curvas de aprendizado específicas para algoritmos de gradiente de política.
    
    Mostra:
    - Recompensas brutas (alta variância típica de PG)
    - Média móvel para ver tendência
    
    Parâmetros:
    -----------
    rewards_dict : Dict[str, List[float]]
        {nome_algoritmo: lista_recompensas}
    window : int, default=100
        Janela para média móvel
    title : str
        Título
    figsize : Tuple[int, int]
        Tamanho da figura
    
    Exemplo:
    --------
    >>> rewards_dict = {
    ...     'REINFORCE': rewards_reinforce,
    ...     'REINFORCE+Baseline': rewards_baseline,
    ...     'Actor-Critic': rewards_ac
    ... }
    >>> plot_policy_gradient_learning(rewards_dict)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Recompensas brutas (mostra variância)
    for name, rewards in rewards_dict.items():
        ax1.plot(rewards, alpha=0.4, linewidth=0.5, label=f'{name}')
    
    ax1.set_xlabel('Episódio', fontsize=11)
    ax1.set_ylabel('Recompensa Total', fontsize=11)
    ax1.set_title('Recompensas Brutas (Alta Variância)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Média móvel (mostra convergência)
    for name, rewards in rewards_dict.items():
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, label=name, linewidth=2.5)
    
    ax2.set_xlabel('Episódio', fontsize=11)
    ax2.set_ylabel(f'Recompensa Média (janela={window})', fontsize=11)
    ax2.set_title('Convergência Suavizada', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_variance(rewards_dict: Dict[str, List[float]],
                    window: int = 50,
                    title: str = "Comparação de Variância",
                    figsize: Tuple[int, int] = (12, 5)):
    """
    Compara variância entre diferentes algoritmos de policy gradient.
    
    Mostra que baseline e actor-critic reduzem variância comparado a REINFORCE puro.
    
    Parâmetros:
    -----------
    rewards_dict : Dict[str, List[float]]
        {nome_algoritmo: lista_recompensas}
    window : int, default=50
        Janela para calcular variância móvel
    title : str
        Título
    figsize : Tuple[int, int]
        Tamanho da figura
    
    Exemplo:
    --------
    >>> compare_variance({
    ...     'REINFORCE': rewards_r,
    ...     'REINFORCE+Baseline': rewards_rb
    ... })
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Desvio padrão móvel
    for name, rewards in rewards_dict.items():
        if len(rewards) >= window:
            # Calcula desvio padrão em janelas deslizantes
            stds = []
            for i in range(len(rewards) - window + 1):
                std = np.std(rewards[i:i+window])
                stds.append(std)
            ax1.plot(stds, label=name, linewidth=2)
    
    ax1.set_xlabel('Episódio', fontsize=11)
    ax1.set_ylabel(f'Desvio Padrão (janela={window})', fontsize=11)
    ax1.set_title('Variância ao Longo do Tempo', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Estatísticas resumidas
    algo_names = []
    means = []
    stds = []
    
    for name, rewards in rewards_dict.items():
        algo_names.append(name)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    
    x_pos = np.arange(len(algo_names))
    ax2.bar(x_pos, stds, color=['red', 'orange', 'green'][:len(stds)], alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algo_names, rotation=15, ha='right')
    ax2.set_ylabel('Desvio Padrão Total', fontsize=11)
    ax2.set_title('Variância Total por Algoritmo', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores no topo das barras
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, s + 0.5, f'{s:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# VISUALIZAÇÃO COMBINADA (VALOR + POLÍTICA)
# ============================================================================

def plot_value_and_policy(w: np.ndarray, theta: np.ndarray, gridworld: GridWorld,
                         title: str = "Função Valor e Política",
                         figsize: Tuple[int, int] = None):
    """
    Visualiza função valor V(s,w) e política π(a|s,θ) lado a lado.
    
    Útil para Actor-Critic: mostra o que o Critic aprendeu (valores)
    e o que o Actor aprendeu (política).
    
    Parâmetros:
    -----------
    w : np.ndarray
        Pesos da função valor
    theta : np.ndarray
        Parâmetros da política
    gridworld : GridWorld
        Ambiente
    title : str
        Título geral
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    
    Exemplo:
    --------
    >>> theta, w, _ = actor_critic(gw, n_episodes=1000)
    >>> plot_value_and_policy(w, theta, gw)
    """
    from algorithms_extended import compute_softmax_policy
    
    if figsize is None:
        figsize = (gridworld.cols * 4, gridworld.rows * 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    action_symbols = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}
    
    # Subplot 1: Função Valor
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, gridworld.cols - 0.5)
        ax.set_ylim(gridworld.rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(gridworld.cols))
        ax.set_yticks(range(gridworld.rows))
        ax.grid(True, alpha=0.3)
    
    ax1.set_title('Função Valor V(s,w)', fontsize=12, fontweight='bold')
    ax2.set_title('Política π(a|s,θ)', fontsize=12, fontweight='bold')
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            state_idx = state[0] * gridworld.cols + state[1]
            
            if state in gridworld.walls:
                for ax in [ax1, ax2]:
                    rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor='gray', alpha=0.8)
                    ax.add_patch(rect)
                continue
            
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                for ax in [ax1, ax2]:
                    rect = FancyBboxPatch((col - 0.4, row - 0.4), 0.8, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, alpha=0.6)
                    ax.add_patch(rect)
                    ax.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                           fontsize=14, fontweight='bold')
                continue
            
            # Valor do estado
            value = w[state_idx]
            ax1.text(col, row, f'{value:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')
            
            # Política
            probs = compute_softmax_policy(state, theta, gridworld)
            best_action_idx = np.argmax(probs)
            best_action = gridworld.actions[best_action_idx]
            symbol = action_symbols[best_action]
            ax2.text(col, row, symbol, ha='center', va='center',
                    fontsize=20, fontweight='bold', color='darkgreen')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNÇÃO PARA EXTRAIR TRAJETO DA POLÍTICA
# ============================================================================

def extract_trajectory(theta: np.ndarray, gridworld: GridWorld,
                      initial_state: Tuple[int, int],
                      max_steps: int = 100,
                      deterministic: bool = True) -> List[Tuple[int, int]]:
    """
    Extrai trajetória seguindo a política aprendida.
    
    Parâmetros:
    -----------
    theta : np.ndarray
        Parâmetros da política
    gridworld : GridWorld
        Ambiente
    initial_state : Tuple[int, int]
        Estado inicial
    max_steps : int, default=100
        Máximo de passos
    deterministic : bool, default=True
        Se True, sempre escolhe ação mais provável
        Se False, amostra segundo π
    
    Retorna:
    --------
    List[Tuple[int, int]]
        Lista de estados visitados
    """
    from algorithms_extended import compute_softmax_policy, sample_action_softmax
    
    trajectory = [initial_state]
    state = initial_state
    
    for _ in range(max_steps):
        if gridworld.is_terminal(state):
            break
        
        if deterministic:
            probs = compute_softmax_policy(state, theta, gridworld)
            action_idx = np.argmax(probs)
        else:
            action_idx = sample_action_softmax(state, theta, gridworld)
        
        action = gridworld.actions[action_idx]
        next_state, _ = gridworld.sample_transition(state, action)
        
        trajectory.append(next_state)
        state = next_state
    
    return trajectory
