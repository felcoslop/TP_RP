# Pesos pre-treinados — baixe e coloque nesta pasta

Os arquivos de pesos NAO acompanham o repositorio (sao grandes). Baixe e
salve nesta pasta `weights/` exatamente com os nomes abaixo.

## Para a maneira PADRAO (`--modelo atual`)

| Arquivo | Tamanho | Usado para |
|---|---|---|
| `cityscale_vitb_512_e10.ckpt` | ~1.0 GB | deteccao em screenshots claros de mapas (Google/Bing). Patch 512. |

**Download direto (OneDrive):**
https://1drv.ms/f/c/5f1c3aa3f12af1a2/IgAXyJ8a0YxFTKGKjhWP1d7SAaFLQOYzJqZubJ9Ipjfn44Y?e=J5HokV

Espelho (SAM-Road, dataset CityScale):
https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link

## Para a maneira FINE-TUNED (`--modelo finetuned`)

> NOTA: o fine-tuning NAO foi realizado neste trabalho (sem GPU local; o Colab
> gratuito so deixa treinar ~4 h, insuficiente para a ViT-B). Todos os
> resultados usam o checkpoint pre-treinado acima. Esta secao fica documentada
> para quem quiser gerar o checkpoint depois.

| Arquivo | Tamanho | Usado para |
|---|---|---|
| `finetuned_spacenet_local.ckpt` | ~1.0 GB | satelite bruto/escuro e estradas de terra. Patch 256. |

Este arquivo voce mesmo gera no Google Colab a partir do dataset local —
siga `training/finetune_local.md` e o notebook `training/colab_finetune.ipynb`.
Ao terminar, baixe o melhor checkpoint do seu Drive e salve aqui com este nome.

Enquanto o fine-tune nao estiver pronto, voce pode usar o checkpoint publico
do SpaceNet como substituto (mesmo patch 256): baixe `spacenet_vitb_256_e10.ckpt`
do mesmo link acima, renomeie para `finetuned_spacenet_local.ckpt` e coloque aqui.

## Peso base (apenas para TREINAR, nao para rodar a deteccao)

| Arquivo | Tamanho | Usado para |
|---|---|---|
| `sam_vit_b_01ec64.pth` | ~360 MB | inicializar o encoder no fine-tune (Colab). |

Download (Meta AI, Segment Anything):
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

---

Depois de colocar os arquivos, confira que os nomes batem exatamente. O
`run.py` procura primeiro em `weights/` e depois em `../weights/`.
