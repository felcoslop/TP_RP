# Documentacao Tecnica de Pesos Pre-Treinados e Modelos

Este documento fornece os detalhes tecnicos, especificacoes e o passo a passo completo para o uso dos pesos pre-treinados utilizados pelo pipeline de extracao de malha viaria na Versao Thick/Tiny.

---

## 1. Tabela Geral de Pesos do Projeto

Os seguintes arquivos de pesos sao utilizados pelo modelo e devem ser organizados no diretorio `weights/` localizado na pasta raiz do projeto:

| Nome do Arquivo | Tamanho do Arquivo | Origem dos Pesos | Utilidade no Pipeline |
|---|---|---|---|
| `cityscale_vitb_512_e10.ckpt` | ~1.00 GB (1.054.078.423 bytes) | SAM-Road ajustado no dataset CityScale | Peso principal de deteccao geometrrica e segmentacao de solo usado pelo `run_thick.py`. |
| `spacenet_vitb_256_e10.ckpt` | ~0.99 GB (1.046.803.927 bytes) | SAM-Road ajustado no dataset SpaceNet | Checkpoint alternativo focado em estradas suburbanas e rurais/terra. |
| `sam_vit_b_01ec64.pth` | ~357 MB (375.042.383 bytes) | Meta AI (Segment Anything original) | Pesos base do codificador de imagem (Image Encoder ViT) antes do ajuste de vias. |

---

## 2. Passo a Passo para Download e Configuracao dos Pesos

Siga estes passos para configurar os pesos na sua maquina fisica:

### Passo 1: Criar a Pasta de Pesos
Crie um diretorio chamado `weights` na raiz do seu projeto ou dentro da pasta `versão_1/`:
```bash
mkdir weights
```

### Passo 2: Efetuar o Download dos Arquivos

1. Download do Checkpoint Principal (CityScale ViT-B 512):
   * Link Oficial:
     https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link
   * Salve como: `weights/cityscale_vitb_512_e10.ckpt`

2. Download do Checkpoint Alternativo (SpaceNet ViT-B 256):
   * Link Oficial:
     https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link
   * Salve como: `weights/spacenet_vitb_256_e10.ckpt`

3. Download do Peso Base SAM ViT-B (Meta AI):
   * Link Oficial:
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   * Salve como: `weights/sam_vit_b_01ec64.pth`

### Passo 3: Verificacao dos Arquivos
Certifique-se de que os nomes dos arquivos salvos correspondem exatamente aos listados acima. O script `run_thick.py` busca automaticamente pela pasta `weights/cityscale_vitb_512_e10.ckpt` na execucao padrao.

---

## 3. Especificacoes Tecnicas Profundas de Cada Peso

### 3.1 Checkpoint Ajustado: CityScale ViT-B 512 (`cityscale_vitb_512_e10.ckpt`)
Este e o peso principal de execucao do pipeline. Ele e um checkpoint completo contendo os tensores ajustados (fine-tuned) para imagens de satelite na resolucao de 512x512 pixels.

* Documentacao e Repositorio Original: [SAM-Road (htcr/sam-road)](https://github.com/htcr/sam-road)
* Artigo de Referencia: "SAM-Road: A Segment Anything Model for Road Network Extraction"
* Arquitetura Compativeil: Vision Transformer (ViT-B) como encoder e ConvTranspose2d como decoder de mascaras.
* Epocas de Ajuste: 10 epocas completas.
* Dataset de Treino: CityScale (derivado do Sat2Graph, compreendendo imagens aereas de alta resolucao de 20 cidades dos Estados Unidos e Europa).
* Caracteristicas do Dataset de Treino: Imagens urbanas densas focando em malhas asfaltadas e padroes bem definidos de organizacao metropolitana.
* Estrutura de Tensores (State Dict):
  * `image_encoder.*`: Pesos treinados do codificador de imagem baseados no SAM ViT-B.
  * `map_decoder.*`: Pesos do decodificador geometrico (Ramo B) responsaveis pela predicao da mascara de vias e dos pontos de intersecao.
  * `topo_net.*`: Pesos do decodificador topologico (Ramo C). Nao sao carregados pela Versao Thick/Tiny para poupar processamento.

### 3.2 Checkpoint Ajustado: SpaceNet ViT-B 256 (`spacenet_vitb_256_e10.ckpt`)
Este checkpoint serve como alternativa direta. Ele foi treinado em um patch size menor (256x256) e possui caracteristicas espectrais diferentes.

* Documentacao e Repositorio Original: [SAM-Road (htcr/sam-road)](https://github.com/htcr/sam-road)
* Arquitetura Compativel: ViT-B (img_size=256) + ConvTranspose2d.
* Dataset de Treino: SpaceNet3 (imagens com resolucao de 1.0 metro/pixel obtidas por satelite).
* Caracteristicas do Dataset de Treino: Alta variacao geografica, incluindo estradas rurais, vias suburbanas e terrenos com solo exposto (terra/cascalho).
* Utilidade: Devido a maior variedade de solo exposto presente no SpaceNet, este modelo costuma apresentar thresholds de deteccao mais consistentes ao lidar com estradas de terra finas ou cobertas parcialmente por vegetacao rural.

### 3.3 Peso Base: SAM ViT-B (`sam_vit_b_01ec64.pth`)
Este e o modelo de fundacao (foundation model) publicado pela Meta AI Research no artigo "Segment Anything".

* Documentacao e Repositorio Original: [Segment Anything Model (facebookresearch/segment-anything)](https://github.com/facebookresearch/segment-anything)
* Artigo de Referencia: "Segment Anything" (Kirillov et al., 2023)
* Arquitetura: Vision Transformer Base (ViT-B) com 12 camadas, dimensao de incorporacao de 768 e 12 cabecas de atencao.
* Dataset de Treino: SA-1B (11 milhoes de imagens de alta resolucao e mais de 1 bilhao de mascaras geradas de forma semi-automatica).
* Importancia: Serve como inicializador para os pesos do Image Encoder (`models/encoder.py`). Ele ja possui capacidades consolidadas de extracao de contornos, bordas e formas geometricas, o que reduz drasticamente o tempo e o volume de dados necessarios para o ajuste subsequente em estradas de satelite.

---

## 4. Como os Pesos sao Carregados no Pipeline (`thick_roads.py`)

O script `thick_roads.py` instacia a classe `ThickRoadDetector` que realiza a particao e o carregamento cirurgico das chaves do checkpoint `.ckpt` da seguinte forma:

1. Instanciacao dos Modelos PyTorch:
   ```python
   self.encoder = SAMEncoder(img_size=512).to(self.device).eval()
   self.decoder = GeometryDecoder().to(self.device).eval()
   ```

2. Carregamento e Remapeamento de Chaves:
   Como o checkpoint original contem chaves unificadas para os ramos A, B e C, o script remapeia as chaves para que correspondam de forma estrita a estrutura simplificada local do encoder e decoder:
   ```python
   ckpt = torch.load(path, map_location=self.device)
   sd = ckpt["state_dict"]
   
   # Filtra e remapeia as chaves do Image Encoder do SAM
   enc = {k.replace("image_encoder.", ""): v for k, v in sd.items() if k.startswith("image_encoder.")}
   
   # Filtra e remapeia as chaves do Geometry Decoder (Ramo B)
   dec = {k.replace("map_decoder.", ""): v for k, v in sd.items() if k.startswith("map_decoder.")}
   
   # Carrega os pesos de forma rigorosa
   self.encoder.encoder.load_state_dict(enc, strict=True)
   self.decoder.decoder.load_state_dict(dec, strict=True)
   ```

---

## 5. Requisitos de Normalizacao Espectral

Os pesos do encoder SAM ViT-B exigem estritamente a aplicacao das constantes de normalizacao do ImageNet. A normalizacao e calculada de forma nativa e automatica dentro do `models/encoder.py`:

* Media de Pixel (pixel_mean): `[123.675, 116.28, 103.53]` (RGB, escala 0-255)
* Desvio Padrao de Pixel (pixel_std): `[58.395, 57.12, 57.375]` (RGB, escala 0-255)

Formula de Processamento Interno do Tensor:
```python
x = (x - self.pixel_mean) / self.pixel_std
```

> [!CAUTION]
> A normalizacao espera floats de imagem no intervalo de 0 a 255. Nao alimente a rede com valores previamente normalizados de 0 a 1, pois isso destruira a escala espectral esperada pelos pesos, resultando em previsoes totalmente vazias.
