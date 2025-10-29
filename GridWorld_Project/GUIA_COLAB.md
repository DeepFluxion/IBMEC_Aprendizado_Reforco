# 🚀 Guia Rápido: Google Colab

## ⚡ Setup em 3 Passos (2 minutos)

### Passo 1️⃣: Upload dos Arquivos

Cole esta célula no Colab e execute:

```python
from google.colab import files
uploaded = files.upload()
print(f"✅ {len(uploaded)} arquivo(s) carregado(s)")
```

**👆 Clique em "Escolher arquivos" e selecione:**
- `environment.py`
- `algorithms.py`
- `visualization.py`

---

### Passo 2️⃣: Imports

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from environment import create_classic_gridworld
from algorithms import q_learning, get_greedy_policy
from visualization import visualize_gridworld, plot_learning_curves

print("✅ Pronto!")
```

---

### Passo 3️⃣: Executar

```python
gw = create_classic_gridworld()
Q, rewards = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)
visualize_gridworld(gw, policy=get_greedy_policy(Q, gw))
plot_learning_curves({'Q-Learning': rewards})
```

**🎉 Pronto! Você já está usando RL!**

---

## 📋 Template Completo para Colab

Copie e cole este notebook completo:

```python
# =============================================================================
# REINFORCEMENT LEARNING - GRIDWORLD
# Template Completo para Google Colab
# =============================================================================

# -----------------------------------------------------------------------------
# 1️⃣ UPLOAD DOS ARQUIVOS
# -----------------------------------------------------------------------------
from google.colab import files
print("📤 Faça upload dos 3 arquivos .py")
uploaded = files.upload()

# -----------------------------------------------------------------------------
# 2️⃣ VERIFICAR ARQUIVOS
# -----------------------------------------------------------------------------
import os
arquivos = ['environment.py', 'algorithms.py', 'visualization.py']
ok = all(f in os.listdir('.') for f in arquivos)
print("✅ Arquivos OK" if ok else "❌ Faltam arquivos")

# -----------------------------------------------------------------------------
# 3️⃣ IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from environment import create_classic_gridworld, create_cliff_world
from algorithms import sarsa, q_learning, expected_sarsa, get_greedy_policy
from visualization import (
    visualize_gridworld, 
    plot_learning_curves, 
    compare_algorithms
)

print("✅ Módulos importados!")

# -----------------------------------------------------------------------------
# 4️⃣ CRIAR AMBIENTE
# -----------------------------------------------------------------------------
gw = create_classic_gridworld()
visualize_gridworld(gw, title="GridWorld 4x3")

# -----------------------------------------------------------------------------
# 5️⃣ TREINAR ALGORITMOS
# -----------------------------------------------------------------------------
print("🧠 Treinando...")

Q_q, rewards_q = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1, verbose=True)
Q_s, rewards_s = sarsa(gw, n_episodes=1000, alpha=0.1, epsilon=0.1, verbose=True)
Q_e, rewards_e = expected_sarsa(gw, n_episodes=1000, alpha=0.1, epsilon=0.1, verbose=True)

print("✅ Treinamento concluído!")

# -----------------------------------------------------------------------------
# 6️⃣ VISUALIZAR RESULTADOS
# -----------------------------------------------------------------------------
print("📊 Visualizando...")

# Curvas de aprendizado
plot_learning_curves({
    'Q-Learning': rewards_q,
    'SARSA': rewards_s,
    'Expected SARSA': rewards_e
})

# Políticas
for nome, Q in [('Q-Learning', Q_q), ('SARSA', Q_s), ('Expected SARSA', Q_e)]:
    policy = get_greedy_policy(Q, gw)
    visualize_gridworld(gw, policy=policy, title=f"{nome} - Política")

# Comparação
compare_algorithms({'Q-Learning': Q_q, 'SARSA': Q_s, 'Expected SARSA': Q_e}, gw)

print("🎉 Experimento concluído!")
```

---

## 🎯 Exemplos Rápidos

### Exemplo 1: Q-Learning Básico

```python
from environment import create_classic_gridworld
from algorithms import q_learning, get_greedy_policy
from visualization import visualize_gridworld

gw = create_classic_gridworld()
Q, _ = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)
visualize_gridworld(gw, policy=get_greedy_policy(Q, gw))
```

### Exemplo 2: Comparar 3 Algoritmos

```python
from algorithms import sarsa, q_learning, expected_sarsa
from visualization import plot_learning_curves

gw = create_classic_gridworld()

_, r1 = sarsa(gw, n_episodes=1000)
_, r2 = q_learning(gw, n_episodes=1000)
_, r3 = expected_sarsa(gw, n_episodes=1000)

plot_learning_curves({
    'SARSA': r1,
    'Q-Learning': r2,
    'Expected SARSA': r3
})
```

### Exemplo 3: Cliff World

```python
from environment import create_cliff_world
from algorithms import sarsa, q_learning
from visualization import visualize_gridworld, get_greedy_policy

gw_cliff = create_cliff_world()

Q_s, _ = sarsa(gw_cliff, n_episodes=500)
Q_q, _ = q_learning(gw_cliff, n_episodes=500)

visualize_gridworld(gw_cliff, policy=get_greedy_policy(Q_s, gw_cliff), 
                   title="SARSA - Conservador")
visualize_gridworld(gw_cliff, policy=get_greedy_policy(Q_q, gw_cliff),
                   title="Q-Learning - Agressivo")
```

---

## 💾 Salvar no Google Drive (Persistente)

### Setup Inicial

```python
# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Criar pasta
import os
pasta = '/content/drive/MyDrive/RL_GridWorld'
os.makedirs(pasta, exist_ok=True)

print(f"📁 Pasta: {pasta}")
print("📤 Faça upload dos .py nesta pasta via Google Drive")
```

### Usar Arquivos do Drive

```python
# Adicionar ao path
import sys
sys.path.append('/content/drive/MyDrive/RL_GridWorld')

# Importar normalmente
from environment import create_classic_gridworld
from algorithms import q_learning

print("✅ Importado do Drive!")
```

### Salvar Resultados

```python
import numpy as np

# Salvar Q-table
np.save('/content/drive/MyDrive/RL_GridWorld/Q_qlearning.npy', Q)

# Salvar rewards
np.save('/content/drive/MyDrive/RL_GridWorld/rewards.npy', rewards)

print("💾 Resultados salvos no Drive!")
```

---

## 🐛 Troubleshooting Rápido

### ❌ Erro: "No module named 'environment'"

**Solução:**
```python
# Verificar arquivos
import os
print(os.listdir('.'))

# Se não aparecer, fazer upload novamente
from google.colab import files
uploaded = files.upload()
```

### ❌ Erro: "name 'plt' is not defined"

**Solução:**
```python
import matplotlib.pyplot as plt
%matplotlib inline
```

### ❌ Gráficos não aparecem

**Solução:**
```python
# Sempre no início do notebook
%matplotlib inline
import matplotlib.pyplot as plt
```

### ❌ MC Exploring Starts demora muito

**Solução:**
```python
# Adicionar max_steps
from algorithms import mc_exploring_starts
Q, rewards = mc_exploring_starts(gw, n_episodes=500, max_steps=200)
```

---

## 🎓 Células Úteis

### Célula: Reiniciar Ambiente

```python
# Limpar variáveis e reiniciar
%reset -f

# Reimportar
from environment import create_classic_gridworld
from algorithms import q_learning
from visualization import visualize_gridworld

print("🔄 Ambiente reiniciado!")
```

### Célula: Ver Documentação

```python
# Ver ajuda de qualquer função
help(q_learning)
help(create_classic_gridworld)
```

### Célula: Verificar Versões

```python
import numpy as np
import matplotlib
import sys

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

### Célula: Instalar Dependências (se necessário)

```python
!pip install --upgrade numpy matplotlib
```

---

## 📊 Experimentos Prontos

### Experimento 1: Sensibilidade ao Alpha

```python
alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
results = {}

for alpha in alphas:
    _, rewards = q_learning(gw, n_episodes=500, alpha=alpha)
    results[f'α={alpha}'] = rewards

plot_learning_curves(results, title="Sensibilidade ao Alpha")
```

### Experimento 2: Sensibilidade ao Epsilon

```python
epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
results = {}

for eps in epsilons:
    _, rewards = q_learning(gw, n_episodes=500, epsilon=eps)
    results[f'ε={eps}'] = rewards

plot_learning_curves(results, title="Sensibilidade ao Epsilon")
```

### Experimento 3: TD vs MC

```python
from algorithms import td_zero_prediction, first_visit_mc_prediction

policy = {s: 'L' for s in gw.states if not gw.is_terminal(s)}

V_td = td_zero_prediction(gw, policy, n_episodes=1000)
V_mc = first_visit_mc_prediction(gw, policy, n_episodes=1000)

visualize_gridworld(gw, values=V_td, title="TD(0)")
visualize_gridworld(gw, values=V_mc, title="Monte Carlo")
```

---

## ⚡ Atalhos de Teclado no Colab

| Ação | Atalho |
|------|--------|
| Executar célula | `Ctrl + Enter` |
| Executar e próxima | `Shift + Enter` |
| Nova célula acima | `Ctrl + M A` |
| Nova célula abaixo | `Ctrl + M B` |
| Deletar célula | `Ctrl + M D` |
| Modo comando | `Esc` |
| Modo edição | `Enter` |

---

## 🎯 Checklist

Antes de começar seus experimentos:

- [ ] ✅ Fez upload dos 3 arquivos .py
- [ ] ✅ Executou os imports
- [ ] ✅ Testou criar um ambiente
- [ ] ✅ Treinou pelo menos um algoritmo
- [ ] ✅ Visualizou os resultados

**Tudo OK? Você está pronto! 🚀**

---

## 📚 Links Úteis

- **📖 Tutorial Completo:** [TUTORIAL.md](TUTORIAL.md)
- **🚀 Guia Rápido:** [GUIA_RAPIDO.md](GUIA_RAPIDO.md)
- **🏔️ Cliff World:** [GUIA_CLIFF_WORLD.md](GUIA_CLIFF_WORLD.md)
- **📋 Exemplos:** [EXEMPLOS_NOTEBOOK.md](EXEMPLOS_NOTEBOOK.md)

---

## 💡 Dicas Finais

1. **Salve seu notebook** regularmente (Ctrl+S ou File > Save)
2. **Use comentários** para documentar seus experimentos
3. **Copie o notebook** antes de fazer mudanças grandes
4. **Organize em seções** usando markdown
5. **Exporte resultados** para o Drive para não perder

---

**Desenvolvido para facilitar seu aprendizado de RL! 🎓**

**Dúvidas? Consulte a [documentação completa](README_GITHUB.md)! 📚**
