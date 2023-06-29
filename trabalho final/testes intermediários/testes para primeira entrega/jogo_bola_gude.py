import numpy as np

def coloca_bolinha(TABULEIRO, posicao):
    # Retorna quantas bolinhas o jogador deixou no tabuleiro
    if (posicao == 0):
        return 1 #
    
    TABULEIRO[posicao] = 1 - TABULEIRO[posicao]
    return TABULEIRO[posicao]

def joga_dado():
    return np.random.choice([0, 1, 2, 3, 4, 5], replace=True)

def checa_ganhou_jogo(jogador):
    return jogador == 0

def simula_jogo(qtd_inicial_bolinhas):
    TABULEIRO = [0, 0, 0, 0, 0, 0]
    
    jogador_0 = qtd_inicial_bolinhas 
    jogador_1 = qtd_inicial_bolinhas 
    
    while True:
        jogador_0 -= coloca_bolinha(TABULEIRO, joga_dado())
        if checa_ganhou_jogo(jogador_0):
            return 0
        jogador_1 -= coloca_bolinha(TABULEIRO, joga_dado())
        if checa_ganhou_jogo(jogador_1):
            return 1

def monte_carlo(qtd_inicial_bolinhas, n):
    j0 = 0
    j1 = 0
    for i in range(0, n):
        jogador_vencedor = simula_jogo(qtd_inicial_bolinhas)
        if jogador_vencedor == 0:
            j0 += 1
        else:
            j1 += 1
            
    print(j0/n)
    print(j1/n)

monte_carlo(10, 1000)