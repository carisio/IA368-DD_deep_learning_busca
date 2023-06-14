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

- [Apresentação da leitura](./5%20-%20doc2query/leitura/doc2query.pdf)

- Implementação:

  - BM25 sem expansão: [Jupyter notebook](./5%20-%20doc2query/notebook/Aula5_t5_doc2query_parte1_bm25_sem_expans%C3%A3o_de_documentos.ipynb) / [Colab](https://colab.research.google.com/drive/1HAfJOob7U-uw0a8V6mGKv710bsCPytzE?usp=sharing)
  - Fine-tuning e geração de queries: [Jupyter notebook](./5%20-%20doc2query/notebook/Aula5_t5_doc2query_parte2_fine_tuning.ipynb) / [Colab](https://colab.research.google.com/drive/15IzYQY8Tv_hznmVQcSmnv_sdj311KMgq?usp=sharing)
  - BM25 com expansão: [Jupyter notebook](./5%20-%20doc2query/notebook/Aula5_t5_doc2query_parte3_bm25_com_expans%C3%A3o_de_documentos.ipynb) / [Colab](https://colab.research.google.com/drive/1gFtEVUm8-dzlxREt8YqDvakhNcRHphoS?usp=sharing)
  - Teste 1 - BM25 sem os documentos (apenas doc2query): [Jupyter notebook](./5%20-%20doc2query/notebook/Aula5_t5_doc2query_teste1_bm25_apenas_expans%C3%A3o_de_documento.ipynb) / [Colab](https://colab.research.google.com/drive/1sFOUSjWS2h1GFRuuiFG0Gnu8Ukbc8PkP?usp=sharing)
  - [Apresentação](./5%20-%20doc2query/notebook/apresentacao_notebook_doc2query.pdf)

<br>

## Aula 6. Buscadores Densos: DPR

Projeto: Finetuning de um buscador denso. Treino usando dataset tiny do MS-MARCO e avaliação no TREC-COVID. Comparar resultados com busca exaustiva e aproximada.

- [Apresentação da leitura](./6%20-%20busca%20densa/leitura/dpr.pdf)

- Implementação: 

  - Caderno completo com loss calculada usando produto interno: [Jupyter notebook](./6%20-%20busca%20densa/notebook/Aula6_dense_retriever.ipynb) / [Colab](https://colab.research.google.com/drive/1fJ9Xx4v8eiF0wrbMBw8tGs5JZhX86Fkz?usp=sharing)
  - Teste com loss calculada usando produto interno de vetores normalizados (similaridade de coseno): [Jupyter notebook](./6%20-%20busca%20densa/notebook/Aula6_dense_retriever_vetor_unitarios.ipynb) / [Colab](https://colab.research.google.com/drive/1k0H9k_lW5607MwkwWlckRFtIJiaJbmpd?usp=sharing)
  - [Apresentação](./6%20-%20busca%20densa/notebook/apresentacao_notebook_dense_retriever.pdf)

<br>

## Aula 7. Buscadores Esparsos: SPLADE

Projeto: Implementar a fase de indexação e buscas de um modelo esparso.


- [Apresentação da leitura](./7%20-%20splade/leitura/splade.pdf)

- Implementação: [Colab](https://colab.research.google.com/drive/1tMSYSw6gT90ua6mOqYI4t2gKWUAlqG8P?usp=sharing) / [Jupyter notebook](./7%20-%20splade/notebook/Aula7_SPLADE_refatorado.ipynb) / [Apresentação](/7%20-%20splade/notebook/apresentacao_splade.pdf)

<br>

## Aula 8. InPars: Adaptação de modelos para novas tarefas

Projeto: Gerar dataset para treino de modelos de buscas usando a técnica do InPars e avaliar um modelo reranqueador treinado neste dataset no TREC-COVID.

A ideia é usar um LLM como gerador de queries para documentos do TREC-COVID. A partir daí, gerar exemplos negativos usando o BM25 e treinar um reranqueador para o TREC-COVID.

- [Apresentação da leitura](./8%20-%20inpars/leitura/inpars.pdf)

- Implementação

  - Caderno 1 - geração de query a partir de documentos usando gpt-3.5-turbo: [Jupyter notebook](./8%20-%20inpars/notebook/Aula_8_geracao_queries.ipynb)
  - Caderno 2 - geração de documentos não relevantes para as queries geradas usando BM25: [Colab](https://colab.research.google.com/drive/1tbKKunyvzI0hcl1_vW7iIeKvS35QfRnn?usp=sharing) / [Jupyter notebook](./8%20-%20inpars/notebook/Aula8_parte2_geracao_doc_id_negativo.ipynb)
  - Caderno 3 - fine-tuning e testes na base TREC-COVID: [Colab](https://colab.research.google.com/drive/1EtIk67iZliPdS604FDzeK84Np2yU2pQx?usp=sharing) / [Jupyter notebook](./8%20-%20inpars/notebook/Aula8_inpars_parte3.ipynb)
  - [Apresentação](./8%20-%20inpars/notebook/apresentacao_inpars.pdf)

<br>

## Aula 9. Destilação

Projeto:

O objetivo do exercício desta semana é construir alguns pipelines de busca e analisá-los em termos das seguintes métricas:

- Qualidade dos resultados: nDCG@10
- Latência (seg/query)
- USD por query assumindo utilização "perfeita": assim que terminou de processar uma query, já tem outra para ser processada
- USD/mês para deixar o sistema rodando para poucos usuários (ex: 100 queries/dia)
- Custo de indexação em USD

Iremos avaliar os pipelines no TREC-COVID.

A latência precisa ser menor que 2 segundos por query.

Considerar:

- 1,50 USD/hora por A100 ou 0,21 USD/hora por T4 ou 0,50 USD/hora por V100
- 0,03 USD/hora por CPU core
- 0,005 USD/hora por GB de CPU RAM


Dicas:
- Utilizar modelos de busca "SOTA" já treinados no MS MARCO como parte do pipeline, como o [SPLADE distil](https://huggingface.co/naver/splade-cocondenser-selfdistil) (esparso), [contriever](https://huggingface.co/facebook/contriever-msmarco) (denso), [Colbert-v2](https://github.com/stanford-futuredata/ColBERT) (denso), [miniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (reranker), [monoT5-3B](https://huggingface.co/castorini/monot5-3b-msmarco) (reranker), [doc2query minus-minus](https://github.com/terrierteam/pyterrier_doc2query) (expansão de documentos + filtragem com reranqueador na etapa de indexação)
- Variar parametros como número de documentos retornados em cada estagio. Por exemplo, BM25 retorna 1000 documentos, um modelo denso ou esparso pode reranquea-los, e passar os top 50 para o miniLM/monoT5 fazer um ranqueamento final.

<br>

- [Apresentação da leitura](./9%20-%20qualidade%20vs%20eficiencia/leitura/colbert_v2.pdf)

- Implementação: [Colab](https://colab.research.google.com/drive/1Tvf3fithfbWWqdaU-SSbj2GDQXbhPVk8?usp=sharing) / [Jupyter notebook](./9%20-%20qualidade%20vs%20eficiencia/notebook/Aula9_qualidade_vs_eficiencia.ipynb) / [Apresentação](./9%20-%20qualidade%20vs%20eficiencia/notebook/apresentacao_qualidade_eficiencia.pdf)

<br>

## Aula 10. Multi-document QA: Visconde

Projeto:

Implementar um pipeline multidoc QA: dado uma pergunta do usuário, buscamos em uma grande coleção as passagens mais relevantes e as enviamos para um sistema agregador, que irá gerar uma resposta final.

- Avaliar no dataset do IIRC
- Métrica principal: F1
- Usar o gpt-3.5-turbo como modelo agregador. Limitar dataset de teste para 50 exemplos para economizar.

Dicas:

- Se inspirar no pipeline do [Visconde](https://github.com/neuralmind-ai/visconde)


<br>

- [Apresentação da leitura](./10%20-%20visconde/leitura/visconde.pdf)

- Implementação: [Colab](https://colab.research.google.com/drive/1XS6pAmkDA6TBxpIyC5xQeOmFHV3ayJVH?usp=sharing) / [Jupyter notebook](./10%20-%20visconde/notebook/Aula10_Visconde.ipynb) / [Apresentação](./10%20-%20visconde/notebook/apresentacao_visconde.pdf)


## Trabalho Final - Gerador de surveys

### Primeira entrega:

- [Apresentação](./trabalho%20final/entrega%201/apresentacao_entrega1.pptx)

- [Notebook](./trabalho%20final/entrega%201/TF_gera_texto_se%C3%A7%C3%A3o_GPT3.ipynb)

- [Exemplo de texto gerado](./trabalho%20final/entrega%201/texto%20gerado.pdf) sobre o assunto "Text representation for ranking" (macro-assunto  "Neural Information Retrieval"). A escolha dos assuntos foi inspirada [neste survey](https://arxiv.org/abs/2207.13443).