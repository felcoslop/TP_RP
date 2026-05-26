# Instrucoes para Adicao de Pesos

Esta pasta e destinada para armazenar os arquivos de pesos pre-treinados do modelo. Para rodar a execucao do pipeline, coloque os arquivos nesta pasta com os nomes exatos detalhados a seguir.

---

## Arquivos de Pesos e Nomes Exatos

Você deve baixar e colocar nesta pasta os seguintes arquivos com estas nomenclaturas exatas:

1. **Checkpoint Principal (CityScale ViT-B 512):**
   * Nome exato do arquivo: `cityscale_vitb_512_e10.ckpt`
   * Finalidade: Utilizado por padrao no script run_thick.py.

2. **Checkpoint Alternativo (SpaceNet ViT-B 256):**
   * Nome exato do arquivo: `spacenet_vitb_256_e10.ckpt`
   * Finalidade: Utilizado como alternativa de inferencia.

3. **Modelo Base SAM ViT-B (Meta AI):**
   * Nome exato do arquivo: `sam_vit_b_01ec64.pth`
   * Finalidade: Peso base do codificador (utilizado na inicializacao/treinamento).

---

## Como Obter os Pesos

Consulte o manual de documentacao completo de pesos localizado na raiz do repositorio em `documentacao_pesos.md` para encontrar todos os links de download direto oficiais e alternativos de cada um dos arquivos listados acima.
