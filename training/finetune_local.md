# Fine-tuning dos pesos no Google Colab

Este guia gera o checkpoint `finetuned_spacenet_local.ckpt` usado pela
maneira `--modelo finetuned` do `run.py`. O treino "agrega o dataset aos
pesos": parte do checkpoint publico do SpaceNet e continua o treinamento nas
imagens locais (dataset SpaceNet 3 em `../../data`, ~2.800 tiles).

Por que no Colab: o treino completo do Vision Transformer precisa de GPU. O
Colab oferece GPU T4 de graca. O tempo esperado e de 3 a 8 horas para 8
epocas.

---

## Passo 1 — Preparar os arquivos no seu PC

Gere o zip do dataset (PowerShell, ajuste o caminho se necessario):

```powershell
Compress-Archive -Path "C:\Users\Felipe Costa\Downloads\BACKUP\Downloads\TP_RP_2026_1\data\*" -DestinationPath "$env:USERPROFILE\Desktop\spacenet_samroad.zip"
```

O dataset tem ~0,5 GB, entao o zip sai rapido.

## Passo 2 — Criar a pasta no Google Drive

1. Abra https://drive.google.com
2. Na raiz do **Meu Drive**, crie uma pasta chamada exatamente `samroad_finetune`.
3. Suba para dentro dela estes 4 arquivos:
   - `spacenet_samroad.zip` (do Desktop)
   - `weights/spacenet_vitb_256_e10.ckpt` (~1 GB — baixe conforme `weights/LEIA-ME.md`)
   - `weights/sam_vit_b_01ec64.pth` (~360 MB)
   - `training/cldice_loss.py` (deste projeto)

## Passo 3 — Abrir e rodar o notebook no Colab

1. Abra https://colab.research.google.com (mesma conta Google do Drive).
2. Menu **Arquivo -> Fazer upload de notebook** -> selecione `training/colab_finetune.ipynb`.
3. Menu **Ambiente de execucao -> Alterar o tipo** -> Acelerador: **T4 GPU**.
4. Rode as celulas de cima para baixo (Shift+Enter), conferindo cada saida:

| Celula | O que faz | Conferir |
|---|---|---|
| 1 | clona o repo de treino + instala dependencias | imprime `cuda: True` |
| 2 | monta o Drive (autorize no popup) | lista seus 4 arquivos |
| 3 | inspeciona o caminho de dados esperado | se mostrar outro caminho, ajuste `DEST` na celula 4 |
| 4 | descompacta o dataset + copia os pesos | imprime ~14000 arquivos |
| 5 | (opcional) lista splits | — |
| 6 | escreve o config do fine-tune | — |
| 7 | aplica a loss clDice (conectividade) | imprime "clDice aplicado" |
| 8 | **o treino (3-8 h)** | barras por epoca; checkpoints salvam direto no Drive |
| 9 | lista os checkpoints salvos | — |

Os checkpoints sao salvos **direto no Drive** a cada epoca, entao se o Colab
desconectar voce nao perde o progresso. Para retomar: reinicie o ambiente,
re-rode as celulas 1, 2, 4, 6, 7 e, na celula 8, troque a linha do
`torch.load(...)` para apontar para o ultimo `.ckpt` em
`{DRIVE}/finetune_out/`.

## Passo 4 — Trazer o resultado de volta

1. No Drive, abra `samroad_finetune/finetune_out/`, baixe o checkpoint mais
   recente.
2. Salve em `weights/finetuned_spacenet_local.ckpt` (deste projeto).
3. Rode o gate de avaliacao antes de adotar:
   ```bash
   python eval/run_eval.py --tag finetuned --modelo finetuned --limit 12
   ```
   Compare com a tabela de `eval/RESULTS.md`. Adote so se F1/recall
   melhorarem e suas imagens claras nao regredirem.
4. Use nas suas imagens:
   ```bash
   python run.py "minha_imagem.png" --modelo finetuned
   ```

---

## Notas tecnicas

- **clDice** (Shit et al., CVPR 2021): loss que preserva o esqueleto da
  predicao, atacando diretamente o sintoma "estrada que para na metade". O
  notebook aplica `mask_loss + 0.3 * clDice` no canal de via. Codigo em
  `cldice_loss.py`.
- **Split por cidade**: para medir generalizacao (requisito do enunciado),
  da para treinar com Vegas+Xangai+Paris e validar em Cartum. Ha uma celula
  comentada para isso.
- **Patch 256**: o checkpoint resultante usa patches de 256x256 (nativo do
  SpaceNet). Por isso a maneira `--modelo finetuned` ja roda com `patch 256`
  automaticamente.
