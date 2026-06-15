# Detecção e Reconstrução de Malha Viária a partir de Imagens de Satélite

Sistema que identifica vias em imagens de satélite/mapas, classifica o
pavimento (**asfalto** ou **terra**), sobrepõe o resultado na imagem original
e exporta a malha como **grafo conectado** (vértices = cruzamentos, arestas =
segmentos de via). Trabalho da disciplina de Reconhecimento de Padrões.

Esta é a **versão final, enxuta** — contém apenas o melhor caminho de
processamento validado experimentalmente, sem código de teste ou variações
abandonadas.

---

## Índice rápido

- [Como rodar (duas maneiras)](#como-rodar)
- [O que o sistema faz, passo a passo](#pipeline)
- [Saídas geradas](#saidas)
- [Avaliação quantitativa](#avaliacao)
- [Fine-tuning dos pesos](#finetune) — **não executado** (sem GPU; Colab limita a ~4 h)
- [Estrutura de arquivos](#estrutura) — documentação de cada arquivo em [`ARQUIVOS.md`](ARQUIVOS.md)
- **Relatório Final (PDF)**: [`latex/relatorio.pdf`](latex/relatorio.pdf) (código: [`latex/relatorio.tex`](latex/relatorio.tex))
- **Apresentação do Projeto (PDF)**: [`latex/apresentacao/apresentacao_vias.pdf`](latex/apresentacao/apresentacao_vias.pdf) (código: [`latex/apresentacao/apresentacao_vias.tex`](latex/apresentacao/apresentacao_vias.tex))

---

<a name="como-rodar"></a>
## 1. Como rodar

### Instalação (uma vez)

```bash
pip install -r requirements.txt
```

Baixe os pesos e coloque na pasta `weights/` seguindo
[`documentacao_pesos.md`](documentacao_pesos.md). Download direto do checkpoint
principal (`cityscale_vitb_512_e10.ckpt`, ~1 GB):

> **[ Baixar pesos (Google Drive)](https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link)**

### As duas maneiras de rodar

O sistema funciona com **dois conjuntos de pesos**, escolhidos com `--modelo`.
Cada preset já embute o tamanho de patch correto — você não precisa lembrar.

**Maneira 1 — pesos atuais (padrão).** Melhor para screenshots claros de
mapas (Google/Bing Maps):

```bash
python run.py "imagens_teste/Captura de tela 2026-06-12 135241.png"
```

**Maneira 2 — pesos do fine-tune (Colab).** Melhor para satélite bruto/escuro
e estradas de terra. Requer o checkpoint gerado conforme
[`training/finetune_local.md`](training/finetune_local.md):

```bash
python run.py "minha_imagem.png" --modelo finetuned
```

> **Nota: O fine-tuning não foi realizado neste trabalho.** Temos o dataset
> (SpaceNet 3) e o notebook de treino prontos, mas **não dispomos de GPU local**
> e o **Google Colab gratuito limita a sessão a ~4 h** — insuficiente para uma
> época estável de uma ViT-B (~90 M de parâmetros). Por isso **todos os
> resultados usam os pesos pré-treinados (Maneira 1)**; a Maneira 2 fica como
> caminho documentado para trabalho futuro.

**Toda execução salva, por padrão, a imagem de cada etapa do pipeline** na
subpasta `passos/` (ver seção 3). Use `--sem-passos` para desligar (mais rápido).

### Processar a pasta inteira de uma vez

```bash
python processar_pasta.py imagens_teste
```

Carrega o modelo **uma única vez** e processa todas as imagens da pasta,
salvando cada resultado (com os passos) em `outputs/<nome_da_imagem>/`. Aceita
as mesmas opções (`--modelo`, `--limiar-via`, `--toco-frac`, etc.).

### Opções extras

```bash
python run.py "foto.png" --tta                  # robustez a rotação (4x mais lento)
python run.py "foto.png" --sem-passos           # NÃO salva as imagens de cada etapa
python run.py "foto.png" --limiar-via 0.45       # mais exigente: corta vias fracas (ver abaixo)
python run.py "foto.png" --toco-frac 0.85        # remove traços internos de quarteirão mais longos
python run.py "foto.png" --vias-claras nao       # rejeita vias claras (modo urbano, ver abaixo)
python run.py "foto.png" --sem-preprocessamento # desliga o realce canny_fusion
python run.py "foto.png" --sem-contexto         # desliga as máscaras de contexto
python run.py "foto.png" --saida outputs/teste1 # pasta de saída
```

**`--vias-claras`** controla se linhas **claras/neutras** (branco/cinza-claro)
podem ser via. Em **zoom próximo/urbano** uma rua de verdade é asfalto
**escuro**, então linhas claras (calçadas, corredores, bordas de telhado)
não devem virar rua; em **satélite alto/rural** a estrada de **terra é clara**
e deve ser aceita. Modos: `auto` (padrão — rejeita só em zoom próximo, decidido
pela largura típica do scout), `nao` (sempre rejeita — força modo urbano),
`sim` (nunca rejeita — força modo rural/satélite).

**`--brilho-min N`** (padrão 205) é a **tolerância** do portão: é o brilho
mínimo (0–255) para um pixel claro ser rejeitado como via. **Menor N = rejeita
mais** (menos tolerante, pega cinza-claro também); **maior N = mais tolerante**
(só rejeita quase-branco). Abaixo de ~150 começa a comer ruas reais cinza.
Ex.: `python run.py "foto.png" --vias-claras nao --brilho-min 190`.

**`--limiar-via T`** (padrão 0.30) é a **tolerância da máscara de via**: na
saída da rede (`thick_road_mask.png`, onde via = branco), só vira rua o branco
≥ T. **Maior T corta as linhas fracas** (vias falsas que aparecem cinza-claras
na máscara, mais exigente); **menor T aceita vias mais fracas** (bom para
estradas de terra fracas). Ex.: `python run.py "foto.png" --limiar-via 0.50`.
É a forma direta de "aumentar a tolerância" para tirar detecções fracas.

**Por padrão** é criada a subpasta `passos/` dentro do resultado, com uma
imagem por etapa (`1_original`, `2_preprocessamento_canny`,
`3_probabilidade_modelo`, `4_mascaras_contexto`, `5_probabilidade_suprimida`,
`6_binaria`, `7_grafo_bruto`, `8_grafo_refinado`, `9_overlay_final`) — útil
para entender e apresentar como a detecção é feita. Desligue com `--sem-passos`.

### Funciona em qualquer zoom (vias finas e largas)

O sistema **detecta o zoom automaticamente** e adapta a escala — você não
configura nada. Uma passada rápida ("scout") em 0.5x mede a largura típica das
vias:

- **Zoom alto** (vias largas, imagem próxima): processa em 0.5x + 1.0x.
- **Zoom padrão/distante** (vias finas, estradas de terra estreitas):
  processa em 1.0x + **2.0x**. Esse **upscale 2x** é o que permite a rede
  "enxergar" caminhos de terra que teriam só 1–2 pixels na imagem original.

As duas escalas são fundidas pegando a probabilidade máxima por pixel, o que
preserva tanto as conexões finas quanto as largas.

---

<a name="pipeline"></a>
## 2. O que o sistema faz, passo a passo

```
imagem original ──┬─────────────────────────────► classificação / overlay / contexto
                  │                                        ▲
                  ▼ (só p/ a rede)                         │
         realce canny_fusion                               │
                  │                                        │
                  ▼                                        │
   inferência multi-escala (SAM-Road ViT-B) ──► prob. de via + cruzamentos
                  │                                        │
                  ▼                                        │
   máscaras de contexto suprimem floresta/água/telhado/solo
                  │
                  ▼
   binarização (hysteresis) ──► esqueleto ──► GRAFO (fonte única de verdade)
                  │
                  ▼
   refino do grafo: funde nós, poda pontas, faz PONTES nos buracos (Dijkstra)
                  │
                  ▼
   classificação de pavimento POR ARESTA (asfalto/terra) + suavização
                  │
                  ▼
   overlay desenhado do grafo + JSON/GraphML
```

**1. Realce de bordas (canny_fusion).** Antes da rede, a imagem recebe uma
mistura de 15% do mapa de bordas de Canny, que realça as bordas paralelas
características das vias. **Esse realce alimenta apenas a rede neural** — a
classificação de pavimento, as máscaras e o desenho final usam sempre a
imagem original (bordas sintéticas não podem ser confundidas com terra).

**2. Inferência multi-escala (SAM-Road ViT-B).** Um Vision Transformer
(encoder do Segment Anything da Meta, ajustado para vias) processa a imagem em
janelas de 512×512 com 50% de sobreposição, em duas escalas (ver seção 1),
produzindo um mapa de probabilidade de via e um mapa de cruzamentos.

**3. Máscaras de contexto.** `pipeline/context_filter.py` detecta por
cor/textura/forma quatro tipos de falso positivo e **suprime a probabilidade**
naquelas regiões antes de binarizar:
- **vegetação** (florestas, parques) — índice de verde com correção de véu de
  cor global;
- **água** (rios, mar, lagos) — baixa textura + matiz fria + área grande;
- **telhados/quarteirões (polígonos de bloco)** — um quarteirão é uma **região
  2-D grande** de construção; a rua é **fina**; o carro está **sobre a rua**.
  A máscara marca o **bloco residencial inteiro** (telha, concreto, laje) e
  **recorta as ruas** usando a própria probabilidade do modelo. Assim pega o
  quarteirão todo, **não** marca carros como telhado e reduz vias falsas em
  área residencial;
- **solo exposto/desmatamento** — mesma paleta da estrada de terra, mas
  discriminado pela **forma**: estrada é 1-D (esqueleto longo e fino), clareira
  é 2-D. Suprimido parcialmente (a estrada de terra pode atravessá-lo).

Pixels onde a rede tem alta confiança nunca são suprimidos (ex.: ponte sobre
rio).

**4. Binarização por hysteresis.** Pixels acima de 0.30 viram "semente"; os
acima de 0.04 só entram se conectados a uma semente — isso reconstrói trechos
esmaecidos no meio de vias confiantes. Limiares **absolutos** (não dependem da
imagem); há um regime de sinal fraco para domínios fora do treino e um
guard-rail que retorna "sem vias" em imagens sem estrada.

**5. Grafo como fonte única de verdade.** `pipeline/graph_refine.py`
esqueletiza a máscara e constrói um grafo (nós = junções/pontas, arestas =
polylines). Todas as correções acontecem no grafo:
- **funde** nós duplicados de junções;
- **poda** pontas curtas e pontas que terminam dentro de telhado ("via dentro
  de casa");
- **remove traços internos de quarteirão**: uma estrada com **uma ponta solta**
  (rua sem saída) cuja ponta cai **dentro de um quarteirão** (cercada por
  telhado) e cujo **alcance é menor que 80% do menor lado do retângulo do
  quarteirão** é removida — é sombra/vão/beco, não rua. Estradas que ligam dois
  cruzamentos ou que cruzam o bloco são mantidas. Controlável por `--toco-frac`;
- **faz pontes** nos buracos por **caminho de menor custo (Dijkstra
  direcional)** sobre `custo = 1/(probabilidade) + contexto`. Água e telhado
  bloqueiam; vegetação encarece mas não bloqueia (vias sob copas continuam
  conectáveis pelo sinal residual). Resolve "estrada que para na metade".

> **Detecção de domínio: rural vs urbano.** O sistema decide automaticamente se
> a imagem é **rural** = **muita vegetação (>25%) E malha viária esparsa
> (<1,2% da imagem)**. Vegetação sozinha não basta: uma **favela num vale
> verde** tem 40%+ de verde mas malha **densa** — é urbana. A densidade da via
> separa limpo (rural de verdade ≤0,6%; favela/urbano ≥2%).
>
> Em imagem **rural**, todo o refino (calibrado p/ malha urbana, distorce no
> campo): (a) é **pulado** — vai do grafo bruto direto ao overlay; (b) as
> máscaras de **telhado e solo são desligadas** (não há quarteirões no campo,
> só pintariam campos de vermelho e suprimiriam estrada de terra fina); a
> supressão rural usa só vegetação + água; (c) a classe usa o limiar de terra
> **baixo** (mais terra). Em imagem **urbana/favela**, o refino roda normal,
> telhado/solo ligam e o limiar de terra é **alto** (mais asfalto).
> Constantes: `VEG_SKIP_REFINE`, `ROAD_DENSITY_RURAL`, `TERRA_THR_*`.

**6. Classificação de pavimento por aresta.** Em vez de decidir pixel a pixel
(que causa efeito "zebra"), a classe é decidida **por aresta inteira**:
mediana da evidência de cor/textura ao longo do núcleo da via. Uma
suavização tipo Potts (ICM) faz vias que se continuam compartilharem a classe;
quando há uma troca **real** de pavimento no meio de uma via longa, ela é
detectada e a aresta é dividida com um nó de transição. Por fim, um trecho de
**terra cercado só por asfalto** (ilha isolada) é revertido para asfalto — rua
de terra real é um trecho conectado, não um toco solto no meio do asfalto.
**Terra = laranja, asfalto = azul.**

O limiar terra×asfalto é **adaptativo por domínio** (via fração de vegetação):
terra seca clara (`p_terra` ~0,26) e asfalto cinza (~0,09) ficam próximos nessa
evidência RGB, então um limiar único não serve. Imagem **urbana** (pouco verde)
usa limiar **alto (0,45)** — evita asfalto fino virar laranja; imagem
**rural/verde** usa limiar **baixo (0,20)** — pega estrada de terra. Interpola
entre as âncoras de vegetação (`TERRA_THR_URBANO/RURAL` em `thick_roads.py`).
*(Limitação conhecida: a distinção terra-clara × asfalto-cinza é difícil só com
RGB; a solução robusta é a cabeça de superfície supervisionada do fine-tune.)*

**7. Saídas.** O overlay é desenhado a partir do grafo (espessura = largura
medida por Distance Transform) e o grafo é exportado em JSON e GraphML.

### Fundamentação (estado da arte)

A arquitetura **segmentação → grafo → classificação de material por aresta**
segue o estado da arte de 2024–2025: SAM-Road (CVPRW 2024, base deste
projeto), SAM-Road++ / dataset Global-Scale (CVPR 2025) e pipelines
industriais (Brightearth/LuxCarta). A loss **clDice** (CVPR 2021), usada no
fine-tuning, preserva a conectividade do esqueleto.

---

<a name="saidas"></a>
## 3. Saídas geradas (na pasta `--saida`, padrão `outputs/`)

**Cada execução cria a sua própria subpasta** `outputs/run_AAAAMMDD_HHMMSS/`,
então uma rodada nunca sobrescreve a anterior e a pasta mais recente (pela
data/hora no nome) é sempre o último resultado. No fim da execução o programa
imprime o caminho exato a abrir. Dentro de cada subpasta:

| Arquivo | Conteúdo |
|---|---|
| `thick_overlay.png` | imagem original com as vias desenhadas (laranja = terra, azul = asfalto) |
| `thick_surface.png` | máscara de classes sobre fundo preto |
| `thick_road_mask.png` | mapa de probabilidade de via da rede (cinza) |
| `context_masks.png` | depuração das máscaras (**verde**=vegetação, **azul**=água, **vermelho**=telhado/quarteirão, **amarelo**=solo exposto, **rosa/magenta**=rejeição por brilho) |
| `graph_output.json` | grafo: vértices (x, y, tipo) e arestas (pavimento, largura, comprimento, confiança, polyline) |
| `graph_output.graphml` | mesmo grafo em GraphML (abre no Gephi, networkx, etc.) |

---

<a name="avaliacao"></a>
## 4. Avaliação quantitativa

O harness `eval/` mede precisão/recall/F1 da linha central (buffer de 8 px) e
métricas topológicas contra o ground truth do dataset local (SpaceNet 3 em
`../../data`):

```bash
python eval/run_eval.py --tag atual --modelo atual --limit 12
```

Resultados por tile em `eval/results_<tag>.csv`; histórico e análise em
[`eval/RESULTS.md`](eval/RESULTS.md). Resumo: com o pós-processamento em grafo,
o F1 sobe de **0.751 para 0.909** sobre o método anterior (mesmo modelo, mesmos
tiles), com metade dos falsos positivos.

---

<a name="finetune"></a>
## 5. Fine-tuning dos pesos

A maneira `--modelo finetuned` usa um checkpoint que **você gera** no Google
Colab a partir do dataset local. O passo a passo completo (preparar o Drive,
rodar o notebook, trazer o resultado) está em
[`training/finetune_local.md`](training/finetune_local.md). É o caminho
recomendado para melhorar a detecção em imagens de satélite escuras e em
estradas de terra.

> **Status: não executado.** O dataset e o notebook estão prontos, mas o
> fine-tuning **não foi rodado** por falta de **GPU local** e pelo **limite de
> ~4 h por sessão do Colab gratuito** (insuficiente para treinar a ViT-B de
> forma estável). Os resultados deste trabalho usam os pesos pré-treinados; o
> fine-tuning permanece como trabalho futuro.

---

<a name="estrutura"></a>
## 6. Estrutura de arquivos

```
versão_2/
├── run.py                      # ponto de entrada (as duas maneiras de rodar)
├── requirements.txt            # dependências Python
├── README.md                   # este arquivo
├── ARQUIVOS.md                 # o que cada arquivo faz (documentação detalhada)
├── models/                     # arquiteturas PyTorch
│   ├── encoder.py              # encoder SAM ViT-B
│   └── ramo_b.py               # decoder de geometria (via + cruzamentos)
├── pipeline/                   # lógica do sistema
│   ├── thick_roads.py          # orquestrador (inferência + montagem)
│   ├── context_filter.py       # máscaras de vegetação/água/telhado/solo
│   ├── graph_refine.py         # grafo: pontes, podas, classes por aresta, desenho
│   ├── classify.py             # evidência de terra + medição de largura
│   ├── preprocessing.py        # realce canny_fusion
│   └── export.py               # JSON + GraphML
├── eval/                       # avaliação quantitativa
│   ├── run_eval.py             # P/R/F1 + métricas de grafo
│   ├── make_val_split.py       # gera o split de validação
│   ├── val_split.json          # split versionado (10 tiles/cidade)
│   └── RESULTS.md              # histórico de resultados e análise
├── training/                   # fine-tuning no Colab
│   ├── finetune_local.md       # guia passo a passo
│   ├── colab_finetune.ipynb    # notebook pronto
│   └── cldice_loss.py          # loss topológica (conectividade)
├── weights/                    # pesos (baixar e colocar — ver LEIA-ME.md)
│   └── LEIA-ME.md
├── latex/                      # documentação acadêmica
│   ├── relatorio.tex           # relatório completo (artigo)
│   ├── figuras/                # figuras do relatório (geradas por montar_figuras.py)
│   └── apresentacao/           # apresentação Beamer (apresentacao_vias.tex) + figuras/
├── montar_figuras.py          # gera as figuras do relatório e da apresentação
├── imagens_teste/              # imagens de exemplo
└── outputs/                    # resultados gerados
```

### Relatório e apresentação

- **Relatório (artigo):** [`latex/relatorio.tex`](latex/relatorio.tex) — compile
  com `pdflatex relatorio.tex` (rode 2×). Cobre introdução, revisão
  bibliográfica, metodologia (os 9 passos), resultados por imagem e limitações.
- **Apresentação (Beamer):** [`latex/apresentacao/apresentacao_vias.tex`](latex/apresentacao/apresentacao_vias.tex).
- As figuras de ambos saem de `python montar_figuras.py` (lê `outputs/` e grava
  em `latex/figuras/` e `latex/apresentacao/figuras/`).

Cada arquivo está documentado individualmente em [`ARQUIVOS.md`](ARQUIVOS.md).
