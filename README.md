# Repositório para a disciplina IA368-DD: Deep Learning aplicado a sistemas de buscas
*Leandro Carísio Fernandes*

<br>

## 0. Seleção para aluno especial

Projeto: Foi solicitada a construção de um sistema de recuperação de informação usando o algoritmo BM25, com resultados sendo avaliados na base de dados CISI.

- [Relatório](./0%20-%20selecao%20-%20bm25%20e%20cisi%20collection/README.md)
- Implementação: [Jupyter notebook](./0%20-%20selecao%20-%20bm25%20e%20cisi%20collection/notebook/bm25-cisi.ipynb) / [Colab](https://colab.research.google.com/drive/1au_hUeSkTk5u6d4Se2wZJAZqP-avW510?usp=sharing)

<br> 

## Aula 1. Buscador Simples: Booleano, TF-IDF, BM25

Leitura: Seção 1 do artigo ["Pretrained Transformers for Text Ranking: BERT and Beyond"](https://arxiv.org/abs/2010.06467)

Projeto: (1) Usar o BM25 implementado pelo pyserini para buscar queries no TREC-DL 2020; (2) Implementar um buscador booleano/bag-of-words; (3) Implementar um buscador com TF-IDF; (4) Avaliar implementações 1, 2, e 3 no TREC-DL 2020 e calcular o nDCG@10.

Entregas: 

- [Apresentação da leitura](./1%20-%20bm25%20-%20bow%20-%20tfidf/leitura/capitulo_1.pdf)

- Implementação: [Jupyter notebook](./1%20-%20bm25%20-%20bow%20-%20tfidf/notebook/Aula1_bm25_bow_tfidf.ipynb) / [Colab](https://colab.research.google.com/drive/1hELJYqsvUyja9HPeDzc9FU8okqdIjODE?usp=sharing)

<br>

## Aula 2. Classificador binário: Análise de Sentimento e Ranqueamento

Leitura: Seção 3 (até 3.2.2) do artigo ["Pretrained Transformers for Text Ranking: BERT and Beyond"](https://arxiv.org/abs/2010.06467)

Projeto: Reranqueamento usando um modelo estilo-BERT com o treinamento no dataset do MS MARCO e avaliação no TREC-DL 2020

- [Apresentação da leitura](./2%20-%20classificador%20binario%20-%20reranking%20com%20minilm/leitura/capitulo_3.pdf)

- Implementação: [Jupyter notebook](./2%20-%20classificador%20binario%20-%20reranking%20com%20minilm/notebook/Aula2_classificador_binario_mini_bert.ipynb) / [Colab](https://colab.research.google.com/drive/1Xcz-h7uHpbuKNZLlOrlIqj8Ao4V3iQos?usp=sharing) / [Apresentação](./2%20-%20classificador%20binario%20-%20reranking%20com%20minilm/notebook/apresentacao_notebook.pdf)

<br>

## Aula 3. Aplicar LLM's Zero e Few-shot (aplicação escolhida pelo aluno)

Leitura: [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

Projeto: (1) Escolher uma tarefa para resolver de maneira zero ou few-shot. (2) É importante ter uma função de avaliação da qualidade das respostas do modelo few-shot. (3) É possível criar um pequeno dataset de teste manualmente. (4) Usar a API do LLAMA ou do ChatGPT (gpt-3.5-turbo).

- [Apresentação da leitura](./3%20-%20few-shot/leitura/language_models_are_few_shot_learners.pdf)

- Implementação: [Jupyter notebook](./3%20-%20few-shot/notebook/Aula3_llm_zero_few_shot.ipynb) / [Apresentação](./3%20-%20few-shot/notebook/apresentacao_notebook_classificacao_resenhas.pdf)

<br>

## Aula 4. Transformer avançado: Implementação e treinamento (modelagem de linguagem)

Leitura: [A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) ou [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Projeto: Treinar um modelo de linguagem em dados em português e avaliar o modelo usando perplexidade.

- [Apresentação da leitura](./4%20-%20modelo%20linguagem%20em%20pt/leitura/language_models_are_unsupervised_multitask_learners.pdf)

- Implementação: [Jupyter notebook](./4%20-%20modelo%20linguagem%20em%20pt/notebook/Aula4_modelo_linguagem_portugues.ipynb) / [Colab](https://colab.research.google.com/drive/1GETPSW3nxapGHnJKsSQgsCPSnDU4fmnq?usp=sharing) / [Apresentação](./4%20-%20modelo%20linguagem%20em%20pt/notebook/apresentacao_notebook_modelo_linguagem_pt.pdf)

<br>

## Aula 5. Modelo seq2seq: T5 para expansão de documentos (doc2query)

Projeto: Treinar um modelo seq2seq (a partir do T5-base) na tarefa de expansão de documentos.

<br>

- [Apresentação da leitura](./5%20-%20doc2query/leitura/doc2query.pdf)

- Implementação: Jupyter notebook / [Colab](https://colab.research.google.com/drive/1HAfJOob7U-uw0a8V6mGKv710bsCPytzE?usp=sharing) / [Apresentação](./5%20-%20doc2query/notebook/apresentacao_notebook_doc2query.pdf)

## Aula 6. Buscadores Densos: DPR

Projeto:

<br>

## Aula 7. Buscadores Esparsos: SPLADE

Projeto:

<br>

## Aula 8. InPars: Adaptação de modelos para novas tarefas

Projeto:

<br>

## Aula 9. Destilação

Projeto:

<br>

## Aula 10. Multi-document QA: Visconde

Projeto:

<br>

