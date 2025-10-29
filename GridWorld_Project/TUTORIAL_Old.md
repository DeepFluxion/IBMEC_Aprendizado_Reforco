# Tutorial: Usando os Módulos de Reinforcement Learning

## 📚 Índice
1. [Visão Geral](#visão-geral)
2. [Estrutura dos Módulos](#estrutura-dos-módulos)
3. [Criando Ambientes GridWorld](#criando-ambientes-gridworld)
4. [Criando Políticas](#criando-políticas)
5. [Algoritmos de Predição](#algoritmos-de-predição)
6. [Algoritmos de Controle](#algoritmos-de-controle)
7. [Visualização de Resultados](#visualização-de-resultados)
8. [Exemplos Completos](#exemplos-completos)
9. [Usando help()](#usando-help)

---

## 1. Visão Geral

Este tutorial mostra como usar os três módulos Python para experimentos de Aprendizado por Reforço:

- **`environment.py`**: Criação e manipulação de ambientes GridWorld
- **`algorithms.py`**: Implementações de algoritmos de RL
- **`visualization.py`**: Visualização de resultados

### Instalação

Coloque os três arquivos `.py` na mesma pasta do seu notebook Jupyter.

### Importação Básica

```python
# Importar módulos
import numpy as np
import matplotlib.pyplot as plt

# Importar classes e funções principais
from environment import (
    GridWorld,
    create_classic_gridworld,
    create_custom_gridworld,
    print_gridworld_info
)

from algorithms import (
    # Predição
    td_zero_prediction,
    first_visit_mc_prediction,
    # Controle
    sarsa,
    q_learning,
    expected_sarsa,
    first_visit_mc_control,
    mc_exploring_starts,
    # Auxiliares
    get_greedy_policy
)

from visualization import (
    visualize_gridworld,
    visualize_q_values,
    visualize_q_table_detailed,
    plot_learning_curves,
    plot_value_evolution,
    plot_value_heatmap,
    plot_q_value_heatmap,
    compare_algorithms,
    print_q_table
)
```

---

## 2. Estrutura dos Módulos

### 📦 `environment.py`

**Classe Principal:**
- `GridWorld`: Ambiente de grid com transições estocásticas

**Funções de Criação:**
- `create_classic_gridworld()`: Grid 4x3 clássico do Russell & Norvig
- `create_custom_gridworld()`: Cria grid personalizado
- `create_cliff_world()`: Ambiente Cliff World

**Função Auxiliar:**
- `print_gridworld_info()`: Imprime informações do ambiente

### 🧠 `algorithms.py`

**Algoritmos de Predição:**
- `td_zero_prediction()`: TD(0)
- `first_visit_mc_prediction()`: Monte Carlo First-Visit

**Algoritmos de Controle:**
- `sarsa()`: SARSA (on-policy)
- `q_learning()`: Q-Learning (off-policy)
- `expected_sarsa()`: Expected SARSA
- `first_visit_mc_control()`: MC Control
- `mc_exploring_starts()`: MC Exploring Starts

**Funções Auxiliares:**
- `get_greedy_policy()`: Extrai política gulosa de Q
- `epsilon_greedy_action()`: Escolhe ação ε-greedy

### 📊 `visualization.py`

**Visualização de Grid:**
- `visualize_gridworld()`: Visualiza grid com valores/política
- `visualize_q_values()`: Mostra valores máximos por estado
- `visualize_q_table_detailed()`: Mostra Q(s,a) de todas as ações

**Curvas de Aprendizado:**
- `plot_learning_curves()`: Plota curvas de aprendizado
- `plot_value_evolution()`: Evolução de valores ao longo do tempo

**Heatmaps:**
- `plot_value_heatmap()`: Heatmap de valores V(s)
- `plot_q_value_heatmap()`: Heatmap de Q-values

**Análise:**
- `compare_algorithms()`: Compara múltiplos algoritmos
- `print_q_table()`: Imprime tabela Q formatada

---

## 3. Criando Ambientes GridWorld

### 3.1 GridWorld Clássico 4x3

```python
# Criar ambiente clássico
gw = create_classic_gridworld()

# Visualizar
visualize_gridworld(gw, title="GridWorld 4x3 Clássico")

# Imprimir informações
print_gridworld_info(gw)
```

**Saída:**
```
======================================================================
INFORMAÇÕES DO GRIDWORLD
======================================================================
Dimensões: 3 linhas x 4 colunas
Total de estados: 11
Paredes: 1
Estados terminais: 2
Fator de desconto (γ): 0.9
Ruído: 0.2
Living reward: -0.04
Ações disponíveis: ['N', 'S', 'L', 'O']

Paredes: [(1, 1)]

Estados terminais:
  (0, 3): reward = 1.0
  (1, 3): reward = -1.0
======================================================================
```

### 3.2 GridWorld Personalizado

```python
# Criar grid 5x5 personalizado
gw_custom = create_custom_gridworld(
    rows=5,
    cols=5,
    walls=[(1, 1), (1, 2), (2, 1), (2, 2)],  # Bloco de paredes
    terminals={(0, 4): 10.0, (4, 0): -10.0},  # Dois terminais
    gamma=0.95,
    noise=0.1,
    living_reward=-0.1
)

visualize_gridworld(gw_custom, title="GridWorld Personalizado 5x5")
print_gridworld_info(gw_custom)
```

### 3.3 Cliff World

```python
# Criar Cliff World
gw_cliff = create_cliff_world()

visualize_gridworld(gw_cliff, title="Cliff World")
```

### 3.4 Criando Manualmente

```python
# Criar grid vazio
gw_manual = GridWorld(rows=4, cols=6, gamma=0.9, noise=0.2)

# Adicionar paredes
gw_manual.set_wall(1, 2)
gw_manual.set_wall(1, 3)
gw_manual.set_wall(2, 2)

# Adicionar terminais
gw_manual.set_terminal(0, 5, 5.0)   # Goal
gw_manual.set_terminal(3, 5, -5.0)  # Trap

# Configurar living reward
gw_manual.living_reward = -0.05

visualize_gridworld(gw_manual, title="GridWorld Manual")
```

---

## 4. Criando Políticas

### 4.1 Política Aleatória

```python
def create_random_policy(gridworld):
    """Cria política completamente aleatória."""
    policy = {}
    for state in gridworld.states:
        if not gridworld.is_terminal(state):
            policy[state] = np.random.choice(gridworld.actions)
    return policy

# Usar
policy_random = create_random_policy(gw)
visualize_gridworld(gw, policy=policy_random, title="Política Aleatória")
```

### 4.2 Política Fixa

```python
def create_fixed_policy(gridworld, action='N'):
    """Cria política que sempre escolhe a mesma ação."""
    policy = {}
    for state in gridworld.states:
        if not gridworld.is_terminal(state):
            policy[state] = action
    return policy

# Sempre ir para Norte
policy_north = create_fixed_policy(gw, 'N')
visualize_gridworld(gw, policy=policy_north, title="Política: Sempre Norte")
```

### 4.3 Política Customizada

```python
def create_custom_policy(gridworld):
    """Cria política específica para o problema."""
    policy = {}
    
    # Exemplo: estratégia para alcançar (0,3)
    policy[(0, 0)] = 'L'  # Ir para direita
    policy[(0, 1)] = 'L'
    policy[(0, 2)] = 'L'
    policy[(1, 0)] = 'N'  # Subir
    policy[(1, 2)] = 'N'
    policy[(2, 0)] = 'N'
    policy[(2, 1)] = 'L'
    policy[(2, 2)] = 'L'
    policy[(2, 3)] = 'N'
    
    return policy

policy_custom = create_custom_policy(gw)
visualize_gridworld(gw, policy=policy_custom, title="Política Customizada")
```

### 4.4 Política Gulosa de Q-values

```python
# Após treinar um algoritmo de controle
Q, _ = q_learning(gw, n_episodes=1000)

# Extrair política gulosa
policy_greedy = get_greedy_policy(Q, gw)

visualize_gridworld(gw, policy=policy_greedy, title="Política Gulosa (Q-Learning)")
```

---

## 5. Algoritmos de Predição

### 5.1 TD(0) - Temporal Difference

```python
# Criar ambiente e política
gw = create_classic_gridworld()
policy = create_fixed_policy(gw, 'N')

# Executar TD(0)
V_td = td_zero_prediction(
    gridworld=gw,
    policy=policy,
    n_episodes=1000,
    alpha=0.1,
    verbose=True
)

# Visualizar resultados
visualize_gridworld(gw, values=V_td, title="TD(0) - Valores V(s)")
plot_value_heatmap(V_td, gw, title="TD(0) - Heatmap")
```

### 5.2 Monte Carlo First-Visit

```python
# Executar Monte Carlo
V_mc = first_visit_mc_prediction(
    gridworld=gw,
    policy=policy,
    n_episodes=1000,
    alpha=0.1,
    verbose=True
)

# Visualizar
visualize_gridworld(gw, values=V_mc, title="MC - Valores V(s)")
plot_value_heatmap(V_mc, gw, title="Monte Carlo - Heatmap")
```

### 5.3 Comparando TD(0) vs Monte Carlo

```python
# Treinar ambos
V_td = td_zero_prediction(gw, policy, n_episodes=1000, alpha=0.1)
V_mc = first_visit_mc_prediction(gw, policy, n_episodes=1000, alpha=0.1)

# Comparar valores
print("Comparação TD(0) vs Monte Carlo:")
print("="*60)
print(f"{'Estado':<15} {'TD(0)':<15} {'MC':<15} {'Diferença':<15}")
print("-"*60)

for state in gw.states:
    if not gw.is_terminal(state) and state not in gw.walls:
        diff = abs(V_td[state] - V_mc[state])
        print(f"{str(state):<15} {V_td[state]:<15.4f} {V_mc[state]:<15.4f} {diff:<15.4f}")
```

### 5.4 Monitorando Evolução dos Valores

```python
# Salvar valores periodicamente
value_history = []
policy = create_fixed_policy(gw, 'L')

# Treinar salvando valores a cada 100 episódios
for checkpoint in range(0, 1000, 100):
    V = td_zero_prediction(gw, policy, n_episodes=100, alpha=0.1)
    value_history.append(V.copy())

# Plotar evolução
states_to_monitor = [(0, 0), (1, 0), (2, 0), (0, 2)]
plot_value_evolution(value_history, states_to_monitor,
                    title="Evolução de V(s) ao longo do treinamento")
```

---

## 6. Algoritmos de Controle

### 6.1 SARSA (On-Policy)

```python
# Treinar SARSA
Q_sarsa, rewards_sarsa = sarsa(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    verbose=True
)

# Extrair política
policy_sarsa = get_greedy_policy(Q_sarsa, gw)

# Visualizar
visualize_gridworld(gw, policy=policy_sarsa, title="SARSA - Política Aprendida")
visualize_q_values(Q_sarsa, gw, title="SARSA - Valores Q")
visualize_q_table_detailed(Q_sarsa, gw, title="SARSA - Q-Values Detalhados")
```

### 6.2 Q-Learning (Off-Policy)

```python
# Treinar Q-Learning
Q_qlearning, rewards_qlearning = q_learning(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    verbose=True
)

# Visualizar
policy_qlearning = get_greedy_policy(Q_qlearning, gw)
visualize_gridworld(gw, policy=policy_qlearning, title="Q-Learning - Política")
visualize_q_values(Q_qlearning, gw, title="Q-Learning - Valores Q")
```

### 6.3 Expected SARSA

```python
# Treinar Expected SARSA
Q_expected, rewards_expected = expected_sarsa(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    verbose=True
)

# Visualizar
policy_expected = get_greedy_policy(Q_expected, gw)
visualize_gridworld(gw, policy=policy_expected, title="Expected SARSA - Política")
visualize_q_values(Q_expected, gw, title="Expected SARSA - Valores Q")
```

### 6.4 Monte Carlo Control

```python
# Treinar MC Control
Q_mc, rewards_mc = first_visit_mc_control(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    epsilon=0.1,
    verbose=True
)

# Visualizar
policy_mc = get_greedy_policy(Q_mc, gw)
visualize_gridworld(gw, policy=policy_mc, title="MC Control - Política")
```

### 6.5 Monte Carlo Exploring Starts

```python
# Treinar MC ES
Q_es, rewards_es = mc_exploring_starts(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    max_steps=500,  # Importante: evita loops infinitos
    verbose=True
)

# Visualizar
policy_es = get_greedy_policy(Q_es, gw)
visualize_gridworld(gw, policy=policy_es, title="MC Exploring Starts - Política")
```

---

## 7. Visualização de Resultados

### 7.1 Visualizar Grid Básico

```python
# Apenas o grid
visualize_gridworld(gw, title="GridWorld Vazio")

# Grid com valores
visualize_gridworld(gw, values=V_td, title="Com Valores V(s)")

# Grid com política
visualize_gridworld(gw, policy=policy_sarsa, title="Com Política")

# Grid com valores E política
visualize_gridworld(gw, values=V_td, policy=policy_sarsa,
                   title="Valores + Política")
```

### 7.2 Visualizar Q-Values

```python
# Valores máximos por estado
visualize_q_values(Q_sarsa, gw, title="SARSA - Max Q-Values")

# Q-values detalhados (todas as ações)
visualize_q_table_detailed(Q_sarsa, gw, title="SARSA - Q(s,a) Detalhado")

# Imprimir tabela Q no console
print_q_table(Q_sarsa, gw, title="SARSA - Tabela Q Completa")
```

### 7.3 Curvas de Aprendizado

```python
# Treinar múltiplos algoritmos
Q_s, rewards_s = sarsa(gw, n_episodes=1000)
Q_q, rewards_q = q_learning(gw, n_episodes=1000)
Q_e, rewards_e = expected_sarsa(gw, n_episodes=1000)

# Plotar curvas
plot_learning_curves({
    'SARSA': rewards_s,
    'Q-Learning': rewards_q,
    'Expected SARSA': rewards_e
}, window=100, title="Comparação de Algoritmos")
```

### 7.4 Heatmaps de Valores

```python
# Heatmap de V(s) - Predição
plot_value_heatmap(V_td, gw, title="TD(0) - Heatmap de Valores")

# Heatmap de max Q(s,a) - Controle
plot_q_value_heatmap(Q_qlearning, gw, title="Q-Learning - Max Q-Values")

# Heatmap para ação específica
plot_q_value_heatmap(Q_qlearning, gw, action='N',
                    title="Q-Learning - Q(s, Norte)")
```

### 7.5 Comparar Algoritmos

```python
# Comparar valores aprendidos
compare_algorithms({
    'SARSA': Q_sarsa,
    'Q-Learning': Q_qlearning,
    'Expected SARSA': Q_expected
}, gw)
```

**Saída:**
```
================================================================================
COMPARAÇÃO DE VALORES APRENDIDOS
================================================================================

Estado         SARSA               Q-Learning          Expected SARSA      
--------------------------------------------------------------------------------
(0, 0)         0.8123              0.8956              0.8542              
(0, 1)         0.9012              0.9234              0.9087              
...

ESTATÍSTICAS RESUMIDAS:
--------------------------------------------------------------------------------

SARSA:
  Valor médio:   0.4523
  Valor máximo:  0.9456
  Valor mínimo:  -0.5234
  Desvio padrão: 0.3421

Q-Learning:
  Valor médio:   0.5012
  Valor máximo:  0.9678
  Valor mínimo:  -0.4123
  Desvio padrão: 0.3256
```

---

## 8. Exemplos Completos

### 8.1 Experimento Completo: Comparação de Algoritmos TD

```python
# ============================================
# CONFIGURAÇÃO
# ============================================
from environment import create_classic_gridworld, print_gridworld_info
from algorithms import sarsa, q_learning, expected_sarsa
from visualization import (
    visualize_gridworld,
    visualize_q_values,
    plot_learning_curves,
    compare_algorithms
)

# Criar ambiente
gw = create_classic_gridworld()
print_gridworld_info(gw)
visualize_gridworld(gw, title="Ambiente: GridWorld 4x3")

# Parâmetros
PARAMS = {
    'n_episodes': 1000,
    'alpha': 0.1,
    'gamma': 0.9,
    'epsilon': 0.1
}

# ============================================
# TREINAMENTO
# ============================================
print("\n→ Treinando SARSA...")
Q_sarsa, rewards_sarsa = sarsa(gw, **PARAMS, verbose=True)

print("\n→ Treinando Q-Learning...")
Q_qlearning, rewards_qlearning = q_learning(gw, **PARAMS, verbose=True)

print("\n→ Treinando Expected SARSA...")
Q_expected, rewards_expected = expected_sarsa(gw, **PARAMS, verbose=True)

# ============================================
# VISUALIZAÇÃO
# ============================================
# Curvas de aprendizado
plot_learning_curves({
    'SARSA': rewards_sarsa,
    'Q-Learning': rewards_qlearning,
    'Expected SARSA': rewards_expected
}, title="Comparação de Convergência")

# Políticas aprendidas
for name, Q in [('SARSA', Q_sarsa), ('Q-Learning', Q_qlearning),
                ('Expected SARSA', Q_expected)]:
    policy = get_greedy_policy(Q, gw)
    visualize_gridworld(gw, policy=policy, title=f"{name} - Política")
    visualize_q_values(Q, gw, title=f"{name} - Valores Q")

# Comparação numérica
compare_algorithms({
    'SARSA': Q_sarsa,
    'Q-Learning': Q_qlearning,
    'Expected SARSA': Q_expected
}, gw)

print("\n✓ Experimento concluído!")
```

### 8.2 Experimento: Sensibilidade ao α

```python
# Testar diferentes valores de alpha
alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
results = {}

print("Testando sensibilidade ao parâmetro α:")
print("="*60)

for alpha in alphas:
    print(f"\n→ Treinando com α = {alpha}...")
    Q, rewards = q_learning(
        gw,
        n_episodes=500,
        alpha=alpha,
        gamma=0.9,
        epsilon=0.1
    )
    
    results[f'α={alpha}'] = rewards
    
    # Calcular valor médio final
    values = []
    for state in gw.states:
        if not gw.is_terminal(state) and state not in gw.walls:
            state_idx = state[0] * gw.cols + state[1]
            values.append(np.max(Q[state_idx]))
    
    avg_value = np.mean(values)
    print(f"   Valor médio final: {avg_value:.4f}")

# Plotar comparação
plot_learning_curves(results, window=50,
                    title="Sensibilidade ao parâmetro α (Q-Learning)")
```

### 8.3 Experimento: TD vs Monte Carlo

```python
# Criar política de teste
policy = create_fixed_policy(gw, 'L')

# Treinar ambos
print("→ Treinando TD(0)...")
V_td = td_zero_prediction(gw, policy, n_episodes=1000, alpha=0.1, verbose=True)

print("\n→ Treinando Monte Carlo...")
V_mc = first_visit_mc_prediction(gw, policy, n_episodes=1000, alpha=0.1, verbose=True)

# Visualizar lado a lado
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# TD(0)
plot_value_heatmap(V_td, gw, title="TD(0) - Valores V(s)")

# Monte Carlo
plot_value_heatmap(V_mc, gw, title="Monte Carlo - Valores V(s)")

# Comparar numericamente
print("\n" + "="*60)
print("COMPARAÇÃO TD(0) vs MONTE CARLO")
print("="*60)
print(f"{'Estado':<15} {'TD(0)':<15} {'MC':<15} {'|Diferença|':<15}")
print("-"*60)

for state in gw.states:
    if not gw.is_terminal(state) and state not in gw.walls:
        diff = abs(V_td[state] - V_mc[state])
        print(f"{str(state):<15} {V_td[state]:<15.4f} {V_mc[state]:<15.4f} {diff:<15.6f}")

# Estatísticas
td_values = [V_td[s] for s in gw.states if not gw.is_terminal(s) and s not in gw.walls]
mc_values = [V_mc[s] for s in gw.states if not gw.is_terminal(s) and s not in gw.walls]

print("\nESTATÍSTICAS:")
print(f"TD(0) - Média: {np.mean(td_values):.4f}, Desvio: {np.std(td_values):.4f}")
print(f"MC    - Média: {np.mean(mc_values):.4f}, Desvio: {np.std(mc_values):.4f}")
```

### 8.4 Experimento: Grid Personalizado

```python
# Criar ambiente personalizado
gw_custom = create_custom_gridworld(
    rows=6,
    cols=6,
    walls=[(2, 2), (2, 3), (3, 2), (3, 3)],
    terminals={(0, 5): 10.0, (5, 0): -10.0},
    gamma=0.95,
    noise=0.1,
    living_reward=-0.1
)

print_gridworld_info(gw_custom)
visualize_gridworld(gw_custom, title="Grid Personalizado 6x6")

# Treinar Q-Learning
Q, rewards = q_learning(
    gw_custom,
    n_episodes=2000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
    verbose=True
)

# Visualizar resultados
policy = get_greedy_policy(Q, gw_custom)
visualize_gridworld(gw_custom, policy=policy, title="Política Aprendida - Grid 6x6")
visualize_q_table_detailed(Q, gw_custom, title="Q-Values - Grid 6x6")

# Plotar curva de aprendizado
plot_learning_curves({'Q-Learning': rewards}, window=100,
                    title="Aprendizado no Grid 6x6")
```

---

## 9. Usando help()

Todos os módulos, classes e funções possuem documentação completa que pode ser acessada com `help()`.

### 9.1 Ajuda para Módulos

```python
# Documentação do módulo inteiro
help(environment)
help(algorithms)
help(visualization)
```

### 9.2 Ajuda para Classes

```python
from environment import GridWorld

# Documentação da classe GridWorld
help(GridWorld)

# Métodos específicos
help(GridWorld.sample_transition)
help(GridWorld.set_terminal)
```

### 9.3 Ajuda para Funções

```python
from algorithms import sarsa, q_learning
from visualization import plot_learning_curves

# Documentação de algoritmos
help(sarsa)
help(q_learning)

# Documentação de visualização
help(plot_learning_curves)
help(visualize_q_values)
```

### 9.4 Exemplo de Uso do help()

```python
# Ver documentação do SARSA
help(sarsa)
```

**Saída:**
```
Help on function sarsa in module algorithms:

sarsa(gridworld: environment.GridWorld, n_episodes: int = 1000, 
      alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, 
      q_init: float = 0.0, initial_state: Tuple[int, int] = None, 
      verbose: bool = False) -> Tuple[numpy.ndarray, List[float]]
    
    Algoritmo SARSA para controle on-policy.
    
    Aprende Q*(s,a) seguindo política ε-greedy.
    Atualiza Q usando a ação que será realmente tomada.
    
    Fórmula:
    --------
    Q(St, At) ← Q(St, At) + α[Rt+1 + γQ(St+1, At+1) - Q(St, At)]
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        Número de episódios
    alpha : float, default=0.1
        Taxa de aprendizado
        - Valores típicos: 0.01 a 0.5
    ...
```

### 9.5 Listando Funções Disponíveis

```python
# Ver todas as funções de um módulo
import algorithms

# Listar funções públicas (sem _ no início)
public_functions = [name for name in dir(algorithms) if not name.startswith('_')]
print("Funções disponíveis em algorithms:")
for func in public_functions:
    print(f"  - {func}")
```

---

## 10. Dicas e Boas Práticas

### 10.1 Salvando Resultados

```python
# Salvar tabela Q
np.save('Q_sarsa.npy', Q_sarsa)

# Carregar tabela Q
Q_loaded = np.load('Q_sarsa.npy')

# Salvar recompensas
np.save('rewards_sarsa.npy', np.array(rewards_sarsa))
```

### 10.2 Organizando Experimentos

```python
# Criar função para experimento completo
def run_experiment(algorithm_name, algorithm_func, gridworld, params):
    """Executa experimento completo e salva resultados."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {algorithm_name}")
    print(f"{'='*60}")
    
    # Treinar
    Q, rewards = algorithm_func(gridworld, **params, verbose=True)
    
    # Extrair política
    policy = get_greedy_policy(Q, gridworld)
    
    # Visualizar
    visualize_gridworld(gridworld, policy=policy,
                       title=f"{algorithm_name} - Política")
    visualize_q_values(Q, gridworld,
                      title=f"{algorithm_name} - Valores Q")
    
    # Salvar
    np.save(f'Q_{algorithm_name}.npy', Q)
    np.save(f'rewards_{algorithm_name}.npy', np.array(rewards))
    
    print(f"\n✓ {algorithm_name} concluído e salvo!")
    
    return Q, rewards, policy

# Usar
params = {'n_episodes': 1000, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.1}
Q_sarsa, rewards_sarsa, policy_sarsa = run_experiment(
    'SARSA', sarsa, gw, params
)
```

### 10.3 Executando Múltiplas Runs

```python
# Para reduzir variância, execute múltiplas vezes
def run_multiple_experiments(algorithm_func, gridworld, params, n_runs=10):
    """Executa algoritmo múltiplas vezes e retorna média."""
    all_rewards = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")
        _, rewards = algorithm_func(gridworld, **params)
        all_rewards.append(rewards)
    
    # Calcular média e desvio padrão
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    return mean_rewards, std_rewards

# Usar
mean_rewards, std_rewards = run_multiple_experiments(
    q_learning, gw,
    {'n_episodes': 500, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.1},
    n_runs=10
)

# Plotar com intervalo de confiança
plt.figure(figsize=(12, 6))
plt.plot(mean_rewards, label='Média')
plt.fill_between(range(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3, label='±1 Desvio Padrão')
plt.xlabel('Episódio')
plt.ylabel('Recompensa')
plt.title('Q-Learning - 10 Runs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 10.4 Debug e Análise

```python
# Verificar se política faz sentido
def analyze_policy(policy, gridworld):
    """Analisa uma política e mostra estatísticas."""
    print("\nANÁLISE DA POLÍTICA:")
    print("="*60)
    
    # Contar ações
    action_counts = {action: 0 for action in gridworld.actions}
    for state, action in policy.items():
        action_counts[action] += 1
    
    print("Distribuição de ações:")
    for action, count in action_counts.items():
        percentage = (count / len(policy)) * 100
        print(f"  {action}: {count} estados ({percentage:.1f}%)")
    
    # Mostrar política por linha
    print("\nPolítica por linha:")
    for row in range(gridworld.rows):
        print(f"  Linha {row}: ", end='')
        for col in range(gridworld.cols):
            state = (row, col)
            if state in policy:
                print(f"{policy[state]}", end=' ')
            else:
                print("·", end=' ')
        print()

# Usar
analyze_policy(policy_sarsa, gw)
```

---

## 11. Troubleshooting

### Problema: ImportError

```python
# Erro: ModuleNotFoundError: No module named 'environment'

# Solução 1: Verificar se arquivos estão na mesma pasta
import os
print("Arquivos na pasta:", os.listdir('.'))

# Solução 2: Adicionar caminho manualmente
import sys
sys.path.append('/caminho/para/pasta/dos/modulos')

# Solução 3: Usar importação relativa se em um pacote
from .environment import GridWorld
```

### Problema: Valores não convergem

```python
# Possíveis causas e soluções:

# 1. α muito alto → diminuir alpha
Q, _ = sarsa(gw, alpha=0.01)  # Em vez de 0.5

# 2. ε muito alto → diminuir epsilon
Q, _ = sarsa(gw, epsilon=0.05)  # Em vez de 0.3

# 3. Poucos episódios → aumentar n_episodes
Q, _ = sarsa(gw, n_episodes=5000)  # Em vez de 500

# 4. Ambiente muito difícil → ajustar living_reward
gw.living_reward = -0.01  # Em vez de -0.1
```

### Problema: Gráficos não aparecem

```python
# Em notebooks Jupyter, adicionar:
%matplotlib inline

# Ou para gráficos interativos:
%matplotlib notebook

# Ou chamar plt.show() explicitamente:
import matplotlib.pyplot as plt
visualize_gridworld(gw)
plt.show()
```

---

## 12. Conclusão

Este tutorial cobriu:

✅ Estrutura dos três módulos  
✅ Criação de ambientes GridWorld  
✅ Definição de políticas  
✅ Algoritmos de predição e controle  
✅ Visualização de resultados  
✅ Exemplos completos de experimentos  
✅ Uso do sistema de ajuda  

**Próximos passos:**
1. Experimente diferentes ambientes
2. Teste diferentes combinações de parâmetros
3. Compare algoritmos em cenários diversos
4. Implemente suas próprias políticas e heurísticas

**Recursos adicionais:**
- Use `help()` para documentação completa
- Consulte os comentários no código-fonte
- Experimente e aprenda fazendo!

---

**Boa sorte com seus experimentos de Reinforcement Learning! 🚀**
