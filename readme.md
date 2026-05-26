# Manual do Pipeline de Deteccao de Malha Viaria - Versao Thick/Tiny

Este documento contem as instrucoes passo a passo para a instalacao, configuracao, e execucao da Versao Thick/Tiny de extracao e reconstrucao de malhas viarias a partir de imagens de satelite, bem como a documentacao detalhada de todos os pesos pre-treinados utilizados.

---

## 1. O que e o Pipeline Thick/Tiny (`run_thick.py`)

A Versao Thick/Tiny e uma implementacao robusta projetada para detectar com alta precisao vias de todas as larguras (estradas principais largas e caminhos de terra finos) atraves de uma inferencia leve e de alta performance. Diferente do pipeline completo tradicional de inferencia multi-escala lento, o pipeline `run_thick.py` otimiza a performance atraves das seguintes tecnicas:

1. Inferencia Multi-Escala Baseada em Scout: Roda uma inferencia rapida em escala scout (0.5x) para estimar a largura tipica das estradas. Se for detectado alto zoom (vias largas), processa a imagem em 0.5x e 1.0x. Se for detectado zoom padrao, processa em 1.0x e 2.0x. Isso garante que vias finas sejam vistas sem sobrecarregar a CPU.
2. Sem Rastreamento Direcional Caro: Remove a etapa mais lenta do pipeline (o trace direcional pixel-a-pixel por busca angular), substituindo-a inteiramente pelo modulo ultra-rapido de processamento e classificacao morfologica `pipeline/classify.py`.
3. Algoritmo de Pontes Morfologicas (Bridge Gaps): Utiliza binarizacao por hysteresis e dilatacao inteligente para unir trechos descontinuados sob copas de arvores e sombras, mantendo a consistencia topologica da malha.
4. Classificacao Semantica de Pavimento: Utiliza a combinacao da textura local (desvio padrao local 7x7 no grayscale) e cor no espaco HSV para segmentar as vias em asfalto (cinza escuro) ou terra (marrom/laranja) em milissegundos.
5. Poda Inteligente de Stubs (Dangling Stubs): Remove dinamicamente ramos sem saida menores que o limite aceitavel de pixels de forma iterativa, limpando ruidos e falsos positivos do grafo final.

---

## 2. Passo a Passo de Execucao (Como rodar)

### Requisitos do Sistema
* Python 3.8 ou superior (Recomendado Python 3.10 ou 3.11)
* Sistema Operacional: Windows, Linux ou macOS
* O pipeline e compativel com CPU e GPU (CUDA). Se uma GPU NVIDIA estiver disponivel, o PyTorch a utilizara automaticamente.

### Passo 1: Instalar Dependencias
Navegue ate a pasta `versão_1` e instale as dependencias necessarias executando no seu terminal:

```bash
pip install -r requirements.txt
```

As dependencias principais incluem:
* torch, torchvision: Motores de deep learning para rodar o encoder e o decoder do SAM.
* segment-anything: Biblioteca oficial da Meta AI para a arquitetura do Vision Transformer (ViT).
* opencv-python-headless: Biblioteca principal de processamento de imagens e visao computacional.
* networkx: Estrutura de dados para construcao e manipulacao de grafos.
* scikit-image: Algoritmos de processamento morfologico e cientifico de imagem, incluindo a esqueletizacao.

### Passo 2: Organizar os Pesos Pre-Treinados
Crie uma pasta chamada `weights` na raiz do projeto (ou dentro da pasta `versão_1`) e coloque os arquivos de pesos baixados la dentro.
Os arquivos de pesos necessarios sao detalhados na secao 3 deste documento.

### Passo 3: Executar a Inferência
Para rodar a deteccao na imagem de teste fornecida, execute o comando a partir da pasta `versão_1`:

```bash
python run_thick.py "Captura de tela 2026-05-12 003536.png"
```

---

## 3. Documentacao dos Pesos Pre-Treinados

O pipeline utiliza uma arquitetura baseada no Vision Transformer (ViT-B) herdado do Segment Anything (SAM) da Meta AI. Os pesos estao estruturados em pesos base e checkpoints ajustados (fine-tuned) para malha viaria.

### 3.1 Peso Base: SAM ViT-B (Meta AI Research)
* Arquivo de peso: `sam_vit_b_01ec64.pth`
* Tamanho do arquivo: ~357 MB (375.042.383 bytes)
* Finalidade: Serve como inicializador para o image encoder da rede. Foi treinado originalmente em mais de 11 milhoes de imagens de alta diversidade (SA-1B) para segmentacao automatica de qualquer objeto.
* Link de Download Oficial:
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

### 3.2 Checkpoint Ajustado: CityScale ViT-B 512 (SAM-Road)
* Arquivo de peso: `cityscale_vitb_512_e10.ckpt`
* Tamanho do arquivo: ~1.00 GB (1.054.078.423 bytes)
* Finalidade: Checkpoint principal ajustado especificamente no dataset CityScale (que contem mapeamentos urbanos de 20 cidades). Mapeia imagens de satelite diretamente em malha viaria e localizacao de intersecoes.
* Estrutura interna: O arquivo `.ckpt` contem o dicionario do PyTorch Lightning com a chave `"state_dict"`, que encapsula:
  * `image_encoder.*`: Pesos do encoder SAM ViT-B (carregado pelo pipeline thick).
  * `map_decoder.*`: Pesos do decoder de geometria (carregado pelo pipeline thick para gerar a probabilidade de estradas e o heatmap de keypoints).
  * `topo_net.*`: Pesos do decodificador de conexao geometrica (nao utilizado no pipeline thick para garantir maxima performance).
* Link de Download Oficial:
  https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link

### 3.3 Checkpoint Ajustado: SpaceNet ViT-B 256
* Arquivo de peso: `spacenet_vitb_256_e10.ckpt`
* Tamanho do arquivo: ~0.99 GB (1.046.803.927 bytes)
* Finalidade: Checkpoint alternativo treinado no dataset SpaceNet3 (com imagens de 1.0 metro/pixel de resolucao). Por ter sido treinado com dados de maior variabilidade de terrenos e regioes suburbanas, ele pode apresentar maior robustez ao lidar com vias rurais e estradas de terra.
* Link de Download Oficial:
  https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link

### 3.4 Como os Pesos Funcionam na Execucao (Muito Importante)

Para rodar a execucao do `run_thick.py`, o pipeline utiliza apenas 1 peso de cada vez (por padrao, o `cityscale_vitb_512_e10.ckpt`).

1. Na Execucao (`run_thick.py`):
O modelo precisa carregar apenas um arquivo de pesos para funcionar:
* `cityscale_vitb_512_e10.ckpt` (Padrao): Este arquivo ja contem todos os pesos necessarios unificados dentro dele. Ele carrega tanto o Codificador de Imagem (SAMEncoder) quanto o Decodificador de Geometria (GeometryDecoder) de uma vez so.
* `spacenet_vitb_256_e10.ckpt` (Alternativo): Se voce preferir usar o SpaceNet, passara ele como argumento, e ele substituira completamente o CityScale durante aquela execucao.

2. O papel do terceiro peso (`sam_vit_b_01ec64.pth`):
O peso original da Meta AI (`sam_vit_b_01ec64.pth`) nao e carregado e nem utilizado durante a inferência (execucao do `run_thick.py`).

Ele serve unicamente para duas coisas:
* Fase de Treinamento (Fine-tuning): E usado para inicializar a rede neural com o aprendizado generico da Meta antes de comecar a treinar o modelo do zero com imagens de satelite.
* Fallback de Inicializacao: Se voce fosse criar ou treinar um decodificador customizado do zero, usaria ele para dar o pontape inicial no encoder.

---

## 4. Passo a Passo do Processamento (Como o Pipeline Thick Funciona)

A execucao do pipeline segue uma sequencia logica e modular estruturada em quatro fases principais:

### Fase 1: Pre-processamento e Analise de Zoom (Scout)
1. Leitura e Conversao: A imagem de entrada e lida e convertida para o formato RGB em float32 (escala 0-255).
2. Inferência Scout: Roda uma inferência inicial rápida em escala reduzida (0.5x) para detectar a largura típica das vias presentes na imagem.
3. Decisão de Resolução:
   * Se a largura média for > 16.0 pixels (zoom alto/ruas largas), o modelo rodará as escalas 0.5x e 1.0x.
   * Se a largura média for <= 16.0 pixels (zoom padrão/ruas normais a finas), o modelo rodará em 1.0x e 2.0x para enxergar detalhes finos.

### Fase 2: Inferência Multi-Escala e Fusão de Probabilidade
1. Janela Deslizante (Sliding Window): Divide a imagem em patches de 512x512 pixels com sobreposição de 50% (stride de 256) em cada uma das escalas escolhidas.
2. Predição dos Patches: O encoder SAM e o Geometry Decoder computam a probabilidade de estradas e cruzamentos por pixel na GPU ou CPU.
3. Fusão Máxima: Combina os outputs de ambas as resoluções calculando a probabilidade máxima (`np.maximum`) por pixel para preservar conexões fracas e finas.

### Fase 3: Reconstrução Morfológica e Classificação Semântica
1. Algoritmo Bridge Gaps: Realiza a binarização por hysteresis baseada em sementes fortes (road_max * 0.15) e componentes fracos conectados (road_max * 0.02). Em seguida, dilata os componentes fortes pelo raio médio da via para criar pontes morfológicas em gaps sob árvores ou sombras.
2. Medição de Largura por Transformada de Distância: Calcula a distância geométrica de cada pixel até a borda da via para atribuir a largura física correta de cada ponto da rua.
3. Classificação Asfalto vs Terra:
   * Analisa a textura local (desvio padrão de brilho 7x7) e a cor no espaço de cores HSV.
   * Vias granulares (alto std) e de tom quente (laranja/marrom no HSV) são classificadas como terra.
   * Vias lisas (baixo std) e de tons neutros (cinza/branco) são classificadas como asfalto.
   * Aplica votação majoritária (>= 65% de dominância) para homogeneizar e unificar o pavimento ao longo do trecho conectado da via.
4. Filtro de Blobs: Remove formas não-elongadas (como manchas de telhados e construções isoladas) que possuam razão de aspecto menor que 2.5.

### Fase 4: Poda Morfológica e Geração de Saída
1. Fechamento de Gaps de Conectividade: Executa buscas em leque a partir de extremidades soltas para reconectar cruzamentos e segmentos de ruas descontinuados.
2. Poda Iterativa de Stubs: Rastreia extremidades sem saída (stubs) de forma iterativa e remove ramos com comprimento menor que o limite aceitável de pixels (`min_stub_px`), eliminando ruídos marginais.
3. Desenho de Overlay e Gravação: Gera o overlay final de saída com as vias sobrepostas sobre a imagem bruta (linhas grossas de asfalto em cinza escuro e terra em marrom/laranja) e salva os arquivos finais em `outputs/`.

---

## 5. Estrutura de Arquivos da Versao Thick/Tiny

```
versão_1/
├── run_thick.py               # Ponto de entrada do pipeline thick/tiny
├── requirements.txt           # Lista de dependencias do Python
├── readme.md                  # Este manual de instrucoes
├── Captura de...003536.png    # Imagem de satelite fornecida para testes
├── models/                    # Definicao das arquiteturas PyTorch
│   ├── __init__.py
│   ├── encoder.py             # Wrapper simplificado do SAM ViT-B
│   └── ramo_b.py              # Geometry Decoder para mascaras e keypoints
└── pipeline/                  # Modulos logicos e heuristicas
    ├── __init__.py
    ├── asphalt_cues.py        # Auxiliares de textura e classificacao
    ├── bridge_gaps.py         # Algoritmo de conexao de lacunas morfologicas
    ├── classify.py            # Hysteresis, classificacao HSV, close gaps e stubs
    ├── context_filter.py      # Filtro de rios e caminhos florestais nao-vias
    ├── edge_detection.py      # Reforco de probabilidade por gradiente de borda
    ├── export.py              # Utilitarios de salvamento e exportacao
    ├── preprocess.py          # Utilitarios basicos de imagem
    ├── scene_cleaner.py       # Mascaramento dinamico de areas verdes/telhados
    ├── signal_gates.py        # Supressores dinamicos espectrais
    ├── thick_roads.py         # Motor principal do pipeline thick/tiny
    ├── tolerant_binarize.py   # Binarizacao hysteresis adaptativa
    └── graph_complete.py      # Completador de grafos topologico
```
