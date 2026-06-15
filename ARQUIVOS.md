# Documentação de cada arquivo

Referência de tudo que existe em `versão_2/`. A pasta contém **apenas** o
caminho de processamento final — nada de scripts de teste ou variações
abandonadas.

---

## Raiz

| Arquivo | O que é |
|---|---|
| `run.py` | **Ponto de entrada (uma imagem).** Lê os argumentos da linha de comando, escolhe o preset de pesos (`--modelo atual` ou `--modelo finetuned`), aplica o realce `canny_fusion` (apenas para a rede) e chama o pipeline. Resolve os pesos em `weights/` ou `../weights/`. Salva os passos por padrão. |
| `processar_pasta.py` | **Processa uma pasta inteira.** Carrega o modelo uma única vez e roda todas as imagens da pasta (padrão `imagens_teste/`), salvando cada resultado + passos em `outputs/<nome>/`. Mesmas opções do `run.py`. |
| `montar_figuras.py` | **Gera as figuras do relatório e da apresentação** a partir de `outputs/`: o grid dos 9 passos por imagem (4 em cima, 5 embaixo), o exemplo do dataset (imagem original primeiro) e os passos individuais. Usa leitura/escrita Unicode-safe (caminhos com "ã"). Grava em `latex/figuras/` e `latex/apresentacao/figuras/`. |
| `requirements.txt` | Dependências Python (torch, opencv, scikit-image, networkx, segment-anything). |
| `README.md` | Visão geral, instalação, as duas maneiras de rodar e explicação do pipeline. |
| `ARQUIVOS.md` | Este arquivo. |

---

## `models/` — arquiteturas PyTorch

| Arquivo | O que é |
|---|---|
| `encoder.py` | Wrapper do encoder **SAM ViT-B** (Segment Anything, Meta AI) com `img_size` configurável (512 ou 256). Normaliza a imagem com as constantes do ImageNet e produz o mapa de features. |
| `ramo_b.py` | **Geometry Decoder**: convoluções transpostas que transformam as features em 2 canais — probabilidade de via e heatmap de cruzamentos. Inclui `extract_nodes_nms` (supressão não-máxima para picos de cruzamento). |
| `__init__.py` | Marca o pacote. |

---

## `pipeline/` — lógica do sistema

| Arquivo | O que é | Funções principais |
|---|---|---|
| `thick_roads.py` | **Orquestrador.** Carrega o modelo, roda a inferência multi-escala com scout (decisão automática de zoom + upscale 2x), aplica contexto, binariza, constrói o grafo, classifica e grava as saídas. | `ThickRoadDetector`, `predict_fused`, `run_pipeline`, `run` |
| `context_filter.py` | **Máscaras de contexto.** Detecta vegetação, água, telhado/prédio e solo exposto e suprime a probabilidade dessas regiões antes da binarização. O solo é separado da estrada de terra pela elongação do esqueleto (1-D vs 2-D). | `build_context_masks`, `apply_suppression`, `masks_debug_image` |
| `graph_refine.py` | **Coração do pós-processamento.** Constrói o grafo do esqueleto, funde nós, poda pontas (inclusive em telhados), remove traços internos de quarteirão (`remove_edges_inside_blocks`), faz pontes por Dijkstra direcional, classifica o pavimento por aresta (**mediana** da evidência num disco apertado + limiar de terra **adaptativo por domínio** + suavização ICM + nós de transição) e reverte terra isolada cercada por asfalto (`flip_isolated_terra`), rasteriza e desenha o overlay. | `graph_from_mask`, `refine_graph`, `classify_edges`, `split_surface_transitions`, `icm_smooth`, `flip_isolated_terra`, `remove_edges_inside_blocks`, `draw_graph_overlay`, `graph_stats`, classe `RoadGraph` |
| `classify.py` | **Evidências de baixo nível.** Mede a largura por Distance Transform e calcula a evidência de "terra" por pixel (cor HSV quente + textura granular, com penalizações de branco/cinza calibradas em QA). | `measure_road_widths`, `compute_p_terra_evidence` |
| `preprocessing.py` | **Realce de entrada.** `canny_fusion`: mistura 15% do mapa de bordas de Canny na imagem, realçando bordas paralelas das vias. Alimenta só a rede. | `preprocess_image` |
| `export.py` | **Exportação do grafo.** Simplifica as polylines (Ramer–Douglas–Peucker) e grava JSON estruturado e GraphML. | `save_graph_json`, `save_graph_graphml`, `graph_to_dict`, `rdp_simplify` |
| `__init__.py` | Marca o pacote. |

---

## `eval/` — avaliação quantitativa

| Arquivo | O que é |
|---|---|
| `run_eval.py` | Roda o pipeline nos tiles do split de validação (SpaceNet 3 em `../../data`) e mede precisão/recall/F1 de centerline (buffer 8 px) + métricas de grafo. Gera `results_<tag>.csv`. |
| `make_val_split.py` | Gera `val_split.json` — amostra determinística e estratificada (10 tiles por cidade). Rode de novo só se o dataset mudar. |
| `val_split.json` | O split de validação versionado (para reprodutibilidade). |
| `RESULTS.md` | Histórico de resultados por configuração + análise das descobertas experimentais. |

---

## `training/` — fine-tuning no Colab

| Arquivo | O que é |
|---|---|
| `finetune_local.md` | Guia passo a passo: preparar o Drive, rodar o notebook, trazer o checkpoint de volta e validar. |
| `colab_finetune.ipynb` | Notebook pronto para o Google Colab (clona o repo de treino, descompacta o dataset, aplica clDice, treina, salva no Drive). |
| `cldice_loss.py` | Implementação da loss **clDice** (preserva a conectividade do esqueleto). Usada pelo notebook. |

---

## `weights/` — pesos pré-treinados

| Arquivo | O que é |
|---|---|
| `LEIA-ME.md` | Instruções para baixar e posicionar os pesos (não acompanham o repositório por serem grandes). Lista os links e os nomes exatos esperados. |

Pesos esperados aqui (você baixa e coloca):
- `cityscale_vitb_512_e10.ckpt` — maneira `--modelo atual`.
- `finetuned_spacenet_local.ckpt` — maneira `--modelo finetuned` (você gera no Colab).
- `sam_vit_b_01ec64.pth` — só para treinar.

---

## `latex/` — documentação acadêmica

| Arquivo | O que é |
|---|---|
| `relatorio.tex` | **Relatório completo (artigo)**: resumo, introdução, revisão bibliográfica (SAM-Road/++, clDice, SpaceNet), dados (com a nota de fine-tuning não realizado), metodologia dos 9 passos com a tabela de legenda de cores, resultados por imagem e limitações. Gera `relatorio.pdf`. |
| `figuras/` | figuras do relatório: `dataset_exemplo.png`, `passo_1..9.png`, `contexto_legenda.png` e `resultado_*.png` (geradas por `montar_figuras.py`). |
| `main_template.tex` | template de artigo usado só como referência de estilo. |

## `latex/apresentacao/` — apresentação (Beamer)

| Arquivo | O que é |
|---|---|
| `apresentacao_vias.tex` | apresentação deste projeto (problema, dataset + nota de fine-tuning, pesos com link de download, pipeline passo a passo, detecção de domínio, resultados por imagem). Gera `apresentacao_vias.pdf` (26 slides). |
| `figuras/` | imagens dos slides: passos individuais + `dataset_exemplo.png` + um `grid_<data>.png` por imagem (os 9 passos, 4 em cima e 5 embaixo). |
| `LEIA-ME.md` | como compilar e como regenerar as figuras. |

## `imagens_teste/` — exemplos

Imagens de satélite/mapa para experimentação (screenshots de Google Maps e uma
foto aérea de marina). O sistema será testado também com imagens não vistas.

## Saídas por execução (`outputs/run_<data_hora>/`)

Cada run cria sua subpasta. Com `--passos`, há também `passos/` com 1 imagem de
cada etapa (`1_original` … `9_overlay_final`) — material direto para o
relatório/apresentação.

## `outputs/` — resultados

Pasta onde o `run.py` grava as saídas (ver a tabela de saídas no `README.md`).
Começa vazia.
