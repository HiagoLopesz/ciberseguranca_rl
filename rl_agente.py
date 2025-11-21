import numpy as np # <-- ESSENCIAL!

# ==========================================================
## 1. CRIAÇÃO DO AMBIENTE (CyberEnv)
# A classe que define o ambiente de cibersegurança para o Agente de RL
class CyberEnv: # <-- DEFINIÇÃO NECESSÁRIA!
    # 0 = Normal, 1 = Baixa Ameaça, 2 = Média Ameaça, 3 = Alta Ameaça
    ESTADOS = [0, 1, 2, 3]
    NUM_ESTADOS = len(ESTADOS)
    
    # 0 = Permitir Conexão (PASS), 1 = Bloquear Conexão (BLOCK)
    ACOES = [0, 1]
    NUM_ACOES = len(ACOES)

    def __init__(self):
        # O estado inicial será sempre "Normal" (0)
        self.estado_atual = 0

    # Método 1: Gerar um novo estado (simular um novo log de rede)
    def reset(self):
        # Simula a probabilidade de um ataque começar ou escalar.
        self.estado_atual = np.random.choice(self.ESTADOS, p=[0.7, 0.15, 0.1, 0.05])
        return self.estado_atual

    # Método 2: O agente toma uma ação e o ambiente calcula a recompensa e o novo estado
    def step(self, acao):
        estado_antes = self.estado_atual
        recompensa = 0
        feito = False # Flag para saber se o episódio terminou

        # Lógica de Recompensa
        if (estado_antes == 3 and acao == 1): # ALTA AMEAÇA + BLOQUEAR
            recompensa = +50
            feito = True
        elif (estado_antes == 0 and acao == 0): # NORMAL + PERMITIR
            recompensa = +1
            
        elif (estado_antes == 3 and acao == 0): # ALTA AMEAÇA + PERMITIR
            recompensa = -100
            feito = True
        elif (estado_antes == 0 and acao == 1): # NORMAL + BLOQUEAR
            recompensa = -5
            
        else:
            recompensa = -1

        # Transição para o Próximo Estado
        if estado_antes == 3 and acao == 0:
            proximo_estado = 3 # O ataque persiste/conclui
        elif estado_antes < 3 and acao == 0:
            proximo_estado = estado_antes + 1 # Ameaça pode escalar
        else:
            proximo_estado = self.reset() # Ameaça foi mitigada ou o ambiente se normaliza
            
        return proximo_estado, recompensa, feito

# ==========================================================
## 2. DESENVOLVIMENTO DO AGENTE (Q-Learning)

# Parâmetros de Q-Learning
ALFA = 0.1       # Taxa de Aprendizagem (Learning Rate)
GAMA = 0.95       # Fator de Desconto (Discount Factor)
EPSILON = 1.0    # Taxa de Exploração (Epsilon)
MIN_EPSILON = 0.01 # Mínimo de Exploração
DECAY_RATE = 0.001 # Taxa de Decaimento do Epsilon

# 1. Inicializar o ambiente
env = CyberEnv()

# 2. Inicializar a Tabela Q
q_tabela = np.zeros((env.NUM_ESTADOS, env.NUM_ACOES))

# Função de seleção de ação (Epsilon-Greedy)
def escolher_acao(estado, epsilon):
    # Exploração: com probabilidade epsilon, escolhe uma ação aleatória
    if np.random.random() < epsilon:
        return np.random.randint(env.NUM_ACOES)
    # Explotação: escolhe a melhor ação baseada na Tabela Q (argmax)
    else:
        return np.argmax(q_tabela[estado, :])

# Função Principal de Treinamento
def treinar_agente(num_episodios):
    global EPSILON
    historico_recompensas = []
    
    print("Iniciando Treinamento Q-Learning...")

    for episodio in range(num_episodios):
        estado_atual = env.reset() 
        recompensa_total = 0
        feito = False
        
        while not feito:
            # 1. Escolher Ação
            acao = escolher_acao(estado_atual, EPSILON)
            
            # 2. Executar Ação e Observar
            proximo_estado, recompensa, feito = env.step(acao)
            
            # 3. Atualizar a Tabela Q (Regra de Bellman)
            q_antigo = q_tabela[estado_atual, acao]
            melhor_q_futuro = np.max(q_tabela[proximo_estado, :])
            
            # Fórmula Principal do Q-Learning: Q(s,a) = Q(s,a) + α * [ R + γ * max(Q(s',a')) - Q(s,a) ]
            novo_q = q_antigo + ALFA * (recompensa + GAMA * melhor_q_futuro - q_antigo)
            q_tabela[estado_atual, acao] = novo_q
            
            estado_atual = proximo_estado
            recompensa_total += recompensa

        # 4. Decaimento do Epsilon
        EPSILON = max(MIN_EPSILON, EPSILON - DECAY_RATE)

        historico_recompensas.append(recompensa_total)
        
        if episodio % 100 == 0:
            print(f"Episódio {episodio}: Recompensa Total = {recompensa_total:.2f}, Epsilon = {EPSILON:.2f}")

    print("\nTreinamento Concluído.")
    return historico_recompensas

# ==========================================================
## 3. EXECUÇÃO E ANÁLISE DE DESEMPENHO

# Executar o treinamento
NUM_EPISODIOS = 1000
historico = treinar_agente(NUM_EPISODIOS)

# 1. Imprimir a Tabela Q Final (O Modelo Aprendido)
print("\n## 📋 Tabela Q Final (Conhecimento do Agente) ##")
print("Linhas = Estados (0:Normal a 3:Alta Ameaça), Colunas = Ações (0:Permitir, 1:Bloquear)")
print(q_tabela)

# 2. Análise da Política (O que o agente faria em cada Estado)
print("\n## 🧠 Política Final (Decisões Preditivas) ##")
for i in range(env.NUM_ESTADOS):
    acao_otima = np.argmax(q_tabela[i, :])
    decisao = "BLOQUEAR" if acao_otima == 1 else "PERMITIR"
    print(f"Estado de Ameaça {i}: Decisão Ótima -> {decisao}")

# 3. Análise de Desempenho (Média das recompensas por episódio)
janela = 50 # Janela de média móvel para suavizar o gráfico
media_movel = np.convolve(historico, np.ones(janela)/janela, mode='valid')

print(f"\nRecompensa Média nos Últimos {janela} Episódios: {np.mean(historico[-janela:]):.2f}")

# Exemplo de como você interpretaria a Tabela Q (Para seu Artigo)
print("\n--- Interpretação para o Artigo ---")
q_alta_ameaca = q_tabela[3, :]
print(f"Q(Estado=3, Permitir): {q_alta_ameaca[0]:.2f}")
print(f"Q(Estado=3, Bloquear): {q_alta_ameaca[1]:.2f}")