# Exemplos de Código - Para Copiar e Colar no Jupyter Notebook

## 🚀 CÉLULA 1: Imports Iniciais

```python
# Imports básicos
import numpy as np
import matplotlib.pyplot as plt

# Configuração para notebooks
%matplotlib inline

# Importar módulos de RL
from environment import (
    GridWorld,
    create_classic_gridworld,
    create_custom_gridworld,
    create_cliff_world,
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

print("✓ Módulos importados com sucesso!")
```

---

## 🏗️ CÉLULA 2: Criar Ambiente Clássico

```python
# Criar GridWorld 4x3 clássico
gw = create_classic_gridworld()

# Visualizar
visualize_gridworld(gw, title="GridWorld 4x3 Clássico")

# Informações
print_gridworld_info(gw)
```

---

## 🎯 CÉLULA 3: Experimento Rápido - Q-Learning

```python
# Treinar Q-Learning
print("→ Treinando Q-Learning...")
Q, rewards = q_learning(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    verbose=True
)

# Extrair política
policy = get_greedy_policy(Q, gw)

# Visualizar
visualize_gridworld(gw, policy=policy, title="Q-Learning - Política Aprendida")
visualize_q_values(Q, gw, title="Q-Learning - Valores Q")

print("\n✓ Treinamento concluído!")
```

---

## 📊 CÉLULA 4: Comparar SARSA vs Q-Learning vs Expected SARSA

```python
# Parâmetros
PARAMS = {
    'n_episodes': 1000,
    'alpha': 0.1,
    'gamma': 0.9,
    'epsilon': 0.1
}

# Treinar os três algoritmos
print("→ Treinando SARSA...")
Q_sarsa, rewards_sarsa = sarsa(gw, **PARAMS, verbose=True)

print("\n→ Treinando Q-Learning...")
Q_qlearning, rewards_qlearning = q_learning(gw, **PARAMS, verbose=True)

print("\n→ Treinando Expected SARSA...")
Q_expected, rewards_expected = expected_sarsa(gw, **PARAMS, verbose=True)

# Visualizar curvas de aprendizado
plot_learning_curves({
    'SARSA': rewards_sarsa,
    'Q-Learning': rewards_qlearning,
    'Expected SARSA': rewards_expected
}, window=100, title="Comparação de Algoritmos TD")

# Comparar valores
compare_algorithms({
    'SARSA': Q_sarsa,
    'Q-Learning': Q_qlearning,
    'Expected SARSA': Q_expected
}, gw)

print("\n✓ Comparação concluída!")
```

---

## 🎨 CÉLULA 5: Visualizações Detalhadas

```python
# Visualizar políticas aprendidas
for name, Q in [('SARSA', Q_sarsa), 
                ('Q-Learning', Q_qlearning),
                ('Expected SARSA', Q_expected)]:
    
    policy = get_greedy_policy(Q, gw)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    
    # Política
    visualize_gridworld(gw, policy=policy, title=f"{name} - Política")
    
    # Valores Q
    visualize_q_values(Q, gw, title=f"{name} - Valores Q")
    
    # Q-values detalhados
    visualize_q_table_detailed(Q, gw, title=f"{name} - Q-Values Detalhados")
    
    # Heatmap
    plot_q_value_heatmap(Q, gw, title=f"{name} - Heatmap")
```

---

## 🔬 CÉLULA 6: Avaliação de Política com TD(0)

```python
# Criar política de teste (sempre ir para Leste)
policy_test = {s: 'L' for s in gw.states if not gw.is_terminal(s)}

# Visualizar política
visualize_gridworld(gw, policy=policy_test, title="Política de Teste: Sempre Leste")

# Avaliar com TD(0)
print("\n→ Avaliando política com TD(0)...")
V_td = td_zero_prediction(
    gridworld=gw,
    policy=policy_test,
    n_episodes=1000,
    alpha=0.1,
    verbose=True
)

# Visualizar resultados
visualize_gridworld(gw, values=V_td, policy=policy_test, 
                   title="TD(0) - Valores + Política")
plot_value_heatmap(V_td, gw, title="TD(0) - Heatmap de Valores")

print("\n✓ Avaliação concluída!")
```

---

## 🆚 CÉLULA 7: TD(0) vs Monte Carlo

```python
# Criar política
policy = {s: 'N' for s in gw.states if not gw.is_terminal(s)}

# Treinar TD(0)
print("→ Treinando TD(0)...")
V_td = td_zero_prediction(gw, policy, n_episodes=1000, alpha=0.1, verbose=True)

# Treinar Monte Carlo
print("\n→ Treinando Monte Carlo...")
V_mc = first_visit_mc_prediction(gw, policy, n_episodes=1000, alpha=0.1, verbose=True)

# Comparar visualmente
print("\n→ Visualizando resultados...")
visualize_gridworld(gw, values=V_td, title="TD(0) - Valores V(s)")
visualize_gridworld(gw, values=V_mc, title="Monte Carlo - Valores V(s)")

# Comparar numericamente
print("\n" + "="*60)
print("COMPARAÇÃO TD(0) vs MONTE CARLO")
print("="*60)
print(f"{'Estado':<15} {'TD(0)':<15} {'MC':<15} {'Diferença':<15}")
print("-"*60)

for state in gw.states:
    if not gw.is_terminal(state) and state not in gw.walls:
        diff = abs(V_td[state] - V_mc[state])
        print(f"{str(state):<15} {V_td[state]:<15.4f} {V_mc[state]:<15.4f} {diff:<15.6f}")

print("\n✓ Comparação concluída!")
```

---

## 🔧 CÉLULA 8: Análise de Sensibilidade - Alpha

```python
# Testar diferentes valores de alpha
alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
results_alpha = {}

print("ANÁLISE DE SENSIBILIDADE - PARÂMETRO ALPHA")
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
    
    results_alpha[f'α={alpha}'] = rewards
    
    # Calcular valor médio
    values = []
    for state in gw.states:
        if not gw.is_terminal(state) and state not in gw.walls:
            state_idx = state[0] * gw.cols + state[1]
            values.append(np.max(Q[state_idx]))
    
    avg_value = np.mean(values)
    print(f"   Valor médio final: {avg_value:.4f}")

# Plotar comparação
plot_learning_curves(results_alpha, window=50,
                    title="Sensibilidade ao α (Taxa de Aprendizado)")

print("\n✓ Análise concluída!")
```

---

## 🔧 CÉLULA 9: Análise de Sensibilidade - Epsilon

```python
# Testar diferentes valores de epsilon
epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
results_epsilon = {}

print("ANÁLISE DE SENSIBILIDADE - PARÂMETRO EPSILON")
print("="*60)

for eps in epsilons:
    print(f"\n→ Treinando com ε = {eps}...")
    Q, rewards = q_learning(
        gw,
        n_episodes=500,
        alpha=0.1,
        gamma=0.9,
        epsilon=eps
    )
    
    results_epsilon[f'ε={eps}'] = rewards
    
    # Calcular recompensa média final
    avg_reward_final = np.mean(rewards[-100:])
    print(f"   Recompensa média (últimos 100): {avg_reward_final:.4f}")

# Plotar comparação
plot_learning_curves(results_epsilon, window=50,
                    title="Sensibilidade ao ε (Exploração)")

print("\n✓ Análise concluída!")
```

---

## 🏗️ CÉLULA 10: Grid Personalizado

```python
# Criar grid personalizado 6x6
gw_custom = create_custom_gridworld(
    rows=6,
    cols=6,
    walls=[(2, 2), (2, 3), (3, 2), (3, 3)],  # Bloco central
    terminals={(0, 5): 10.0, (5, 0): -10.0},  # Dois terminais
    gamma=0.95,
    noise=0.1,
    living_reward=-0.1
)

# Visualizar ambiente
visualize_gridworld(gw_custom, title="Grid Personalizado 6x6")
print_gridworld_info(gw_custom)

# Treinar
print("\n→ Treinando Q-Learning no grid customizado...")
Q_custom, rewards_custom = q_learning(
    gw_custom,
    n_episodes=2000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
    verbose=True
)

# Visualizar resultados
policy_custom = get_greedy_policy(Q_custom, gw_custom)
visualize_gridworld(gw_custom, policy=policy_custom, 
                   title="Grid 6x6 - Política Aprendida")
visualize_q_table_detailed(Q_custom, gw_custom, 
                          title="Grid 6x6 - Q-Values Detalhados")

# Curva de aprendizado
plot_learning_curves({'Q-Learning (6x6)': rewards_custom}, 
                    window=100,
                    title="Aprendizado no Grid 6x6")

print("\n✓ Experimento no grid customizado concluído!")
```

---

## 🎓 CÉLULA 11: Verificar Documentação (help)

```python
# Ver documentação de um algoritmo
print("="*60)
print("DOCUMENTAÇÃO DO SARSA")
print("="*60)
help(sarsa)
```

---

## 💾 CÉLULA 12: Salvar e Carregar Resultados

```python
# Salvar resultados
print("→ Salvando resultados...")

np.save('Q_sarsa.npy', Q_sarsa)
np.save('Q_qlearning.npy', Q_qlearning)
np.save('rewards_sarsa.npy', np.array(rewards_sarsa))
np.save('rewards_qlearning.npy', np.array(rewards_qlearning))

print("✓ Resultados salvos!")

# Carregar resultados (em outra sessão)
print("\n→ Carregando resultados...")

Q_sarsa_loaded = np.load('Q_sarsa.npy')
Q_qlearning_loaded = np.load('Q_qlearning.npy')
rewards_sarsa_loaded = np.load('rewards_sarsa.npy')
rewards_qlearning_loaded = np.load('rewards_qlearning.npy')

print("✓ Resultados carregados!")

# Verificar
print(f"\nShape de Q_sarsa: {Q_sarsa_loaded.shape}")
print(f"Número de episódios: {len(rewards_sarsa_loaded)}")
```

---

## 📝 CÉLULA 13: Imprimir Tabela Q Formatada

```python
# Imprimir tabela Q de forma legível
print_q_table(Q_qlearning, gw, title="Q-LEARNING - TABELA Q COMPLETA")
```

---

## 🎯 CÉLULA 14: Monte Carlo Exploring Starts

```python
# Treinar MC Exploring Starts
print("→ Treinando MC Exploring Starts...")
Q_es, rewards_es = mc_exploring_starts(
    gridworld=gw,
    n_episodes=1000,
    alpha=0.1,
    max_steps=500,  # Limite para evitar loops infinitos
    verbose=True
)

# Extrair política
policy_es = get_greedy_policy(Q_es, gw)

# Visualizar
visualize_gridworld(gw, policy=policy_es, 
                   title="MC Exploring Starts - Política")
visualize_q_values(Q_es, gw, 
                  title="MC Exploring Starts - Valores Q")

# Comparar com Q-Learning
plot_learning_curves({
    'MC Exploring Starts': rewards_es,
    'Q-Learning': rewards_qlearning
}, window=100, title="MC Exploring Starts vs Q-Learning")

print("\n✓ Treinamento concluído!")
```

---

## 🔬 CÉLULA 15: Experimento Completo - Múltiplas Runs

```python
# Executar Q-Learning múltiplas vezes para reduzir variância
print("EXPERIMENTO: MÚLTIPLAS RUNS (10x)")
print("="*60)

n_runs = 10
all_rewards = []

for run in range(n_runs):
    print(f"\n→ Run {run+1}/{n_runs}...")
    Q, rewards = q_learning(
        gw,
        n_episodes=500,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    all_rewards.append(rewards)
    print(f"   Recompensa média final: {np.mean(rewards[-50:]):.2f}")

# Calcular média e desvio padrão
all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

# Plotar com intervalo de confiança
plt.figure(figsize=(12, 6))
plt.plot(mean_rewards, label='Média', linewidth=2)
plt.fill_between(range(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3, label='±1 Desvio Padrão')
plt.xlabel('Episódio')
plt.ylabel('Recompensa')
plt.title('Q-Learning - 10 Runs com Intervalo de Confiança')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Experimento de múltiplas runs concluído!")
```

---

## 🌊 CÉLULA 16: Cliff World

```python
# Criar Cliff World
gw_cliff = create_cliff_world()

# Visualizar ambiente
visualize_gridworld(gw_cliff, title="Cliff World")
print_gridworld_info(gw_cliff)

# Treinar SARSA (conservador)
print("\n→ Treinando SARSA (on-policy, conservador)...")
Q_sarsa_cliff, rewards_sarsa_cliff = sarsa(
    gw_cliff,
    n_episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
)

# Treinar Q-Learning (agressivo)
print("\n→ Treinando Q-Learning (off-policy, agressivo)...")
Q_qlearn_cliff, rewards_qlearn_cliff = q_learning(
    gw_cliff,
    n_episodes=500,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
)

# Comparar políticas
policy_sarsa_cliff = get_greedy_policy(Q_sarsa_cliff, gw_cliff)
policy_qlearn_cliff = get_greedy_policy(Q_qlearn_cliff, gw_cliff)

visualize_gridworld(gw_cliff, policy=policy_sarsa_cliff,
                   title="Cliff World - SARSA (Conservador)")
visualize_gridworld(gw_cliff, policy=policy_qlearn_cliff,
                   title="Cliff World - Q-Learning (Agressivo)")

# Comparar curvas
plot_learning_curves({
    'SARSA (Conservador)': rewards_sarsa_cliff,
    'Q-Learning (Agressivo)': rewards_qlearn_cliff
}, window=50, title="Cliff World: SARSA vs Q-Learning")

print("\n✓ Experimento Cliff World concluído!")
print("\nObservação: SARSA tende a ser mais conservador (evita o cliff)")
print("Q-Learning tende a ser mais agressivo (aprende caminho ótimo mas arriscado)")
```

---

## 🎨 CÉLULA 17: Heatmaps de Q-Values por Ação

```python
# Criar heatmaps para cada ação
acoes = ['N', 'S', 'L', 'O']
nomes = ['Norte', 'Sul', 'Leste', 'Oeste']

print("HEATMAPS DE Q-VALUES POR AÇÃO")
print("="*60)

for acao, nome in zip(acoes, nomes):
    plot_q_value_heatmap(Q_qlearning, gw, action=acao,
                        title=f"Q(s, {nome})")
```

---

## 📊 CÉLULA 18: Análise Final Detalhada

```python
# Análise completa dos resultados
print("\n" + "="*70)
print("ANÁLISE FINAL - Q-LEARNING")
print("="*70)

# Estatísticas da tabela Q
print("\n1. ESTATÍSTICAS DA TABELA Q:")
print("-"*70)
print(f"   Valor máximo:  {np.max(Q_qlearning):.4f}")
print(f"   Valor mínimo:  {np.min(Q_qlearning):.4f}")
print(f"   Valor médio:   {np.mean(Q_qlearning):.4f}")
print(f"   Desvio padrão: {np.std(Q_qlearning):.4f}")

# Análise da política
print("\n2. ANÁLISE DA POLÍTICA:")
print("-"*70)

policy_final = get_greedy_policy(Q_qlearning, gw)

action_counts = {'N': 0, 'S': 0, 'L': 0, 'O': 0}
for action in policy_final.values():
    action_counts[action] += 1

print("   Distribuição de ações:")
for action, count in action_counts.items():
    percentage = (count / len(policy_final)) * 100
    print(f"     {action}: {count} estados ({percentage:.1f}%)")

# Análise de convergência
print("\n3. ANÁLISE DE CONVERGÊNCIA:")
print("-"*70)
print(f"   Recompensa inicial (média primeiros 50): {np.mean(rewards_qlearning[:50]):.2f}")
print(f"   Recompensa final (média últimos 50):     {np.mean(rewards_qlearning[-50:]):.2f}")
print(f"   Melhoria:                                {np.mean(rewards_qlearning[-50:]) - np.mean(rewards_qlearning[:50]):.2f}")

# Visualizar política final
print("\n4. POLÍTICA FINAL:")
print("-"*70)
visualize_gridworld(gw, policy=policy_final, 
                   title="Q-Learning - Política Final")

print("\n✓ Análise completa!")
```

---

## 🔄 CÉLULA 19: Template para Seu Próprio Experimento

```python
# ============================================
# SEU EXPERIMENTO PERSONALIZADO
# ============================================

# 1. CONFIGURAÇÃO
print("="*60)
print("MEU EXPERIMENTO CUSTOMIZADO")
print("="*60)

# Criar ambiente (escolha um)
# gw = create_classic_gridworld()
# gw = create_custom_gridworld(rows=5, cols=5, ...)
# gw = create_cliff_world()

# 2. VISUALIZAR AMBIENTE
# visualize_gridworld(gw, title="Meu Ambiente")
# print_gridworld_info(gw)

# 3. DEFINIR PARÂMETROS
params = {
    'n_episodes': 1000,
    'alpha': 0.1,
    'gamma': 0.9,
    'epsilon': 0.1
}

# 4. TREINAR ALGORITMO (escolha um)
# Q, rewards = sarsa(gw, **params, verbose=True)
# Q, rewards = q_learning(gw, **params, verbose=True)
# Q, rewards = expected_sarsa(gw, **params, verbose=True)

# 5. EXTRAIR POLÍTICA
# policy = get_greedy_policy(Q, gw)

# 6. VISUALIZAR RESULTADOS
# visualize_gridworld(gw, policy=policy, title="Política Aprendida")
# visualize_q_values(Q, gw, title="Valores Q")
# plot_learning_curves({'Meu Algoritmo': rewards})

# 7. ANÁLISE
# print_q_table(Q, gw, title="Tabela Q Final")

print("\n✓ Complete as seções acima para seu experimento!")
```

---

## 📚 CÉLULA 20: Links para Documentação

```python
print("="*70)
print("RECURSOS DE AJUDA")
print("="*70)

print("\n1. Ver todas as funções disponíveis:")
print("   import algorithms")
print("   print([f for f in dir(algorithms) if not f.startswith('_')])")

print("\n2. Ver documentação de uma função:")
print("   help(sarsa)")
print("   help(visualize_gridworld)")
print("   help(GridWorld)")

print("\n3. Ver parâmetros de uma função:")
print("   from inspect import signature")
print("   print(signature(q_learning))")

print("\n4. Tutorial completo:")
print("   Leia o arquivo TUTORIAL.md")

print("\n5. Exemplos rápidos:")
print("   Leia o arquivo README.md")

print("\n" + "="*70)
```

---

## ✅ Pronto!

Copie e cole estas células no seu Jupyter Notebook para começar a experimentar!

Cada célula é independente e pode ser executada separadamente.

**Dica:** Execute as células na ordem para um tutorial completo, ou escolha
células específicas para experimentos focados.
