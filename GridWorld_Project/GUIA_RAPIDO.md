# Guia Rápido de Referência - RL GridWorld

## 📦 Arquivos do Projeto

```
├── environment.py          # Ambientes GridWorld
├── algorithms.py           # Algoritmos de RL
├── visualization.py        # Visualização
├── TUTORIAL.md            # Tutorial completo (LEIA PRIMEIRO!)
├── README.md              # Documentação geral
├── EXEMPLOS_NOTEBOOK.md   # Células prontas para copiar
└── GUIA_RAPIDO.md         # Este arquivo
```

---

## ⚡ Quick Start (3 minutos)

```python
# 1. Importar
from environment import create_classic_gridworld
from algorithms import q_learning, get_greedy_policy
from visualization import visualize_gridworld

# 2. Criar ambiente
gw = create_classic_gridworld()

# 3. Treinar
Q, rewards = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# 4. Visualizar
policy = get_greedy_policy(Q, gw)
visualize_gridworld(gw, policy=policy, title="Política")
```

---

## 📖 Documentação Rápida

### Ver ajuda de qualquer função:
```python
help(q_learning)  # Mostra documentação completa
```

### Listar funções disponíveis:
```python
import algorithms
print([f for f in dir(algorithms) if not f.startswith('_')])
```

---

## 🏗️ Criar Ambientes

### Grid Clássico 4x3:
```python
gw = create_classic_gridworld()
```

### Grid Personalizado:
```python
gw = create_custom_gridworld(
    rows=5, cols=5,
    walls=[(1,1), (2,2)],
    terminals={(0,4): 10.0},
    gamma=0.95
)
```

### Cliff World:
```python
gw = create_cliff_world()
```

---

## 🧠 Algoritmos

### Predição (Avaliação de Política):
```python
# TD(0)
V = td_zero_prediction(gw, policy, n_episodes=1000, alpha=0.1)

# Monte Carlo
V = first_visit_mc_prediction(gw, policy, n_episodes=1000, alpha=0.1)
```

### Controle (Busca de Política Ótima):
```python
# SARSA (on-policy)
Q, rewards = sarsa(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# Q-Learning (off-policy)
Q, rewards = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# Expected SARSA
Q, rewards = expected_sarsa(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# Monte Carlo Control
Q, rewards = first_visit_mc_control(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# MC Exploring Starts
Q, rewards = mc_exploring_starts(gw, n_episodes=1000, alpha=0.1, max_steps=500)
```

### Extrair Política:
```python
policy = get_greedy_policy(Q, gw)
```

---

## 📊 Visualização

### Grid com Política/Valores:
```python
visualize_gridworld(gw, values=V, policy=policy, title="Meu Grid")
```

### Q-Values:
```python
# Valores máximos
visualize_q_values(Q, gw, title="Max Q-Values")

# Detalhado (todas as ações)
visualize_q_table_detailed(Q, gw, title="Q-Values Detalhados")

# Imprimir no console
print_q_table(Q, gw, title="Tabela Q")
```

### Curvas de Aprendizado:
```python
plot_learning_curves({
    'SARSA': rewards_sarsa,
    'Q-Learning': rewards_qlearning
}, window=100)
```

### Heatmaps:
```python
# Valores V(s)
plot_value_heatmap(V, gw, title="Heatmap de Valores")

# Q-values
plot_q_value_heatmap(Q, gw, title="Heatmap Q")
plot_q_value_heatmap(Q, gw, action='N', title="Q(s, Norte)")
```

### Comparar Algoritmos:
```python
compare_algorithms({
    'SARSA': Q_sarsa,
    'Q-Learning': Q_qlearning
}, gw)
```

---

## 🎛️ Parâmetros Comuns

### alpha (Taxa de Aprendizado)
- **0.01-0.05**: Lento mas estável
- **0.1**: Padrão, bom equilíbrio
- **0.3-0.5**: Rápido mas pode oscilar

### epsilon (Exploração)
- **0.0**: Sem exploração (só exploitation)
- **0.1**: Padrão, 10% exploração
- **0.2-0.3**: Mais exploração

### gamma (Desconto)
- **0.9**: Padrão, valoriza futuro próximo
- **0.95-0.99**: Mais visionário
- **0.5-0.7**: Mais míope (imediato)

### n_episodes
- **500-1000**: Rápido, teste inicial
- **1000-2000**: Padrão
- **5000+**: Convergência garantida

---

## 🔧 Padrões de Uso

### Experimento Básico:
```python
# 1. Ambiente
gw = create_classic_gridworld()

# 2. Treinar
Q, rewards = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)

# 3. Visualizar
policy = get_greedy_policy(Q, gw)
visualize_gridworld(gw, policy=policy)
plot_learning_curves({'Q-Learning': rewards})
```

### Comparar 2 Algoritmos:
```python
# Treinar
Q1, r1 = sarsa(gw, n_episodes=1000)
Q2, r2 = q_learning(gw, n_episodes=1000)

# Comparar
plot_learning_curves({'SARSA': r1, 'Q-Learning': r2})
compare_algorithms({'SARSA': Q1, 'Q-Learning': Q2}, gw)
```

### Análise de Sensibilidade:
```python
alphas = [0.01, 0.1, 0.5]
results = {}

for alpha in alphas:
    Q, rewards = q_learning(gw, alpha=alpha)
    results[f'α={alpha}'] = rewards

plot_learning_curves(results)
```

---

## 💾 Salvar/Carregar

```python
# Salvar
np.save('Q_table.npy', Q)
np.save('rewards.npy', rewards)

# Carregar
Q = np.load('Q_table.npy')
rewards = np.load('rewards.npy')
```

---

## 🐛 Problemas Comuns

### Módulos não encontrados:
```python
# Verificar arquivos na pasta
import os
print(os.listdir('.'))
# Deve mostrar: environment.py, algorithms.py, visualization.py
```

### Valores não convergem:
```python
# Solução 1: Mais episódios
Q, _ = q_learning(gw, n_episodes=5000)

# Solução 2: Alpha menor
Q, _ = q_learning(gw, alpha=0.05)

# Solução 3: Mais exploração
Q, _ = q_learning(gw, epsilon=0.2)
```

### Gráficos não aparecem:
```python
# Adicionar no início do notebook
%matplotlib inline
```

---

## 📚 Onde Procurar

- **Como criar ambientes?** → `TUTORIAL.md` Seção 3
- **Como usar algoritmos?** → `TUTORIAL.md` Seções 5-6
- **Como visualizar?** → `TUTORIAL.md` Seção 7
- **Exemplos completos?** → `TUTORIAL.md` Seção 8
- **Células prontas?** → `EXEMPLOS_NOTEBOOK.md`
- **Documentação de função?** → `help(nome_funcao)`

---

## 🎯 Checklist para Experimento

- [ ] Importar módulos
- [ ] Criar/carregar ambiente
- [ ] Visualizar ambiente
- [ ] Definir parâmetros
- [ ] Treinar algoritmo(s)
- [ ] Extrair política
- [ ] Visualizar resultados
- [ ] Analisar valores
- [ ] Salvar resultados (opcional)

---

## 🚀 Próximos Passos

1. Execute o Quick Start acima
2. Leia `TUTORIAL.md` por completo
3. Copie células de `EXEMPLOS_NOTEBOOK.md`
4. Experimente diferentes parâmetros
5. Crie seus próprios ambientes
6. Compare algoritmos

---

## 📞 Ajuda

```python
# Ver documentação
help(algorithms)
help(environment)
help(visualization)

# Ver parâmetros de função
help(q_learning)
help(create_custom_gridworld)
```

---

**Criado para fins educacionais - Aprendizado por Reforço**
