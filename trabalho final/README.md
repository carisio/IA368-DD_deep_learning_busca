## Trabalho Final - Gerador de surveys

Autores: Leandro Carísio Fernandes e Gustavo Bartz Guedes

### Primeira entrega:

- [Apresentação](./entrega%201/apresentacao_entrega1.pptx)

- [Notebook](./entrega%201/TF_gera_texto_se%C3%A7%C3%A3o_GPT3.ipynb)

- [Exemplo de texto gerado](./entrega%201/texto%20gerado.pdf) sobre o assunto "Text representation for ranking" (macro-assunto  "Neural Information Retrieval"). A escolha dos assuntos foi inspirada [neste survey](https://arxiv.org/abs/2207.13443).

### Entrega final:

- [Apresentação](./entrega%20final/apresenta%C3%A7%C3%A3o/apresentacao_final.pdf)

- Notebooks (importados do Colab)

  - [1] [Pesquisa artigos e divide o texto em seções](./entrega%20final/notebooks/%5BTF%5D_%5B1%5D_Sugest%C3%A3o_se%C3%A7%C3%B5es.ipynb)
    - Entrada: Query inicial, filtros e estrutura desejada (número de seções e subseções)
    - Saída: Arquivo pickle(survey_xxxx.pkl) contendo um array de seções. Cada uma tem um nome e uma lista de metadados de artigos possivelmente relacionados
  - [2] [Download e extração de artigos](./entrega%20final/notebooks/%5BTF%5D_%5B2%5D_Extra%C3%A7%C3%A3o_de_texto_dos_artigos.ipynb)
    - Entrada: Arquivo pickle gerado na etapa anterior
    - Saída: Arquivo pickle (paper_contents_xxx.pkl) com um mapa cuja chave é o id do artigo e o valor são os metadados do artigo (abstract e title são obrigatórios) e uma propriedade text contendo o texto do artigo. Esse arquivo pode ser enriquecido com um texto de melhor qualidade extraído pelo Grobid. Nesse caso é necessário adicionar uma propriedade text_grobid.
  - [3] [Geração de texto das seções](./entrega%20final/notebooks/%5BTF%5D_%5B3%5D_Gera_textos_das_se%C3%A7%C3%B5es.ipynb)
    - Entrada: Arquivos pickle usados nas etapas anteriores e uma string com a query inicial usada na etapa 1.
    - Saída: Arquivo pickle survey_xxx.pkl preenchido com o texto da seção gerado.

  - Exemplos gerados:
    - Geração semi-automática: [Texto (docx) das seções gerado para o survey usado como referência](./entrega%20final/survey%20de%20referencia/gpt%203.5/sections-survey-ref_grobid_generated_text_specter_gpt-3.5-turbo-0613.docx)
      - ENTRADA 1: [sections-survey-ref.pkl](./entrega%20final/survey%20de%20referencia/sections-survey-ref.pkl)
      - ENTRADA 2: [papers_contents-survey-ref_grobid.pkl](./entrega%20final/survey%20de%20referencia/papers_contents-survey-ref_grobid.pkl)
      - SAÍDA: [sections-survey-ref_generated_text_splade_gpt-3.5-turbo-0613.pkl](./entrega%20final/survey%20de%20referencia/gpt%203.5/sections-survey-ref_generated_text_splade_gpt-3.5-turbo-0613.pkl)
      - AVALIAÇÃO COM BERTScore (Precision: 78.2%, Recall: 81.4%, F1: 79.8%): [caderno](./entrega%20final/notebooks/%5BTF%5D_%5BEval%5D_%5BSURVEY_REF%5D_com_BERTScore.ipynb)
    
    - Geração automática: [Texto gerado para um pipeline completo - 5 seções e 3 subseções por seção](./entrega%20final/pipeline%20completo%20-%20text%20neural%20information%20retrieval/sections-text%20neural%20information%20retrieval-2020-2023-gpt-3.5-turbo-0613_generated_text_specter_gpt-3.5-turbo-0613.docx)
      - ENTRADA 1: [sections-text neural information retrieval-2020-2023-gpt-3.5-turbo-0613.pkl](./entrega%20final/pipeline%20completo%20-%20text%20neural%20information%20retrieval/sections-text%20neural%20information%20retrieval-2020-2023-gpt-3.5-turbo-0613.pkl)
      - ENTRADA 2: [papers_contents-text neural information retrieval-2020-2023-gpt-3.5-turbo-0613.pkl](./entrega%20final/pipeline%20completo%20-%20text%20neural%20information%20retrieval/papers_contents-text%20neural%20information%20retrieval-2020-2023-gpt-3.5-turbo-0613.pkl)
      - SAÍDA: [sections-text neural information retrieval-2020-2023-gpt-3.5-turbo-0613_generated_text_specter_gpt-3.5-turbo-0613.pkl](./entrega%20final/pipeline%20completo%20-%20text%20neural%20information%20retrieval/sections-text%20neural%20information%20retrieval-2020-2023-gpt-3.5-turbo-0613_generated_text_specter_gpt-3.5-turbo-0613.pkl)

- [Artigo](./entrega%20final/texto/Geracao_automatica_de_survey.pdf)
