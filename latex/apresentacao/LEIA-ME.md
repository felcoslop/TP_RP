# Apresentação (Beamer)

- **`apresentacao_vias.tex`** — apresentação deste projeto (malha viária).
  Participantes: Felipe Costa, Igor Falco, Victor Morais. Disciplina:
  Reconhecimento de Padrões.
- `figuras/` — imagens usadas nos slides (geradas pelo pipeline com `--passos`
  + montagem do dataset + grids de passos por imagem).

O **relatório** (artigo completo) fica em `latex/relatorio.tex` (pasta acima).

## Como compilar (gera o PDF)

Precisa de uma distribuição LaTeX (MiKTeX ou TeX Live). Na pasta:

```bash
pdflatex apresentacao_vias.tex
pdflatex apresentacao_vias.tex   # 2x para montar o índice/navegação
```

Resultado: `apresentacao_vias.pdf` (26 slides — termina com um slide por imagem
de teste mostrando os 9 passos, 4 em cima e 5 embaixo).

## Como regenerar as figuras

As figuras (passos individuais, exemplo do dataset e os grids de 9 passos por
imagem) são geradas automaticamente a partir de `outputs/`:

```bash
cd ../..                       # raiz do versão_2
python processar_pasta.py imagens_teste   # roda o pipeline em todas as imagens
python montar_figuras.py                  # monta as figuras do relatório e da apresentação
```

`montar_figuras.py` grava em `latex/figuras/` (relatório) e em
`latex/apresentacao/figuras/` (apresentação), incluindo um `grid_<data>.png`
por imagem. Não é preciso copiar arquivos à mão.
