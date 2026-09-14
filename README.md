# ACC2RR — baseline de frequência respiratória com Polar H10

Este repositório contém coletas do acelerômetro triaxial do Polar H10 em quatro posturas (`em_pe`, `lado`, `sentado`, `supino`) e duas condições respiratórias (`10rpm`, `20rpm`). O script `analyze_acc2rr.py` implementa o primeiro pipeline clássico/interpretável para estimar a taxa respiratória e, principalmente, produzir diagnóstico suficiente para sabermos onde o método funciona ou falha.

## Pipeline implementado

1. **QC dos timestamps**: ordenação, timestamps duplicados, `Δt`, frequência efetiva, jitter, gaps e estimativa de amostras ausentes. Gaps longos (>0,5 s por padrão) **não são interpolados**: usa-se o maior segmento contínuo.
2. **Reamostragem uniforme** na frequência nominal do `metadata.json` (50 Hz nas coletas atuais).
3. **Hampel vetorial conservador**: um outlier detectado em qualquer eixo faz com que o vetor XYZ inteiro daquele instante seja interpolado, preservando coerência espacial.
4. **Referência pela gravidade**: estima-se o vetor mediano de gravidade e projeta-se a dinâmica no plano perpendicular a ele.
5. **Filtro respiratório** Butterworth de 4ª ordem, `0.08–0.80 Hz` por padrão, aplicado com `sosfiltfilt` (zero phase).
6. **PCA triaxial** no sinal já projetado e filtrado; PC1 vira o *respiratory surrogate*.
7. **Dois estimadores independentes**:
   - Welch PSD, com verificação espectral de segundo harmônico;
   - autocorrelação, buscando a primeira recorrência periódica forte.
8. **Consenso harmônico** entre candidatos de PSD e ACF.
9. **Índices de qualidade**: concentração/pico espectral, força da ACF, variância explicada pela PC1, concordância PSD↔ACF, qualidade dos timestamps e relação respiração/movimento.
10. **Janelas móveis** de 30 s, deslocadas a cada 1 s por padrão.
11. **Tracker temporal 1-D (Kalman)** para suavizar a série sem apagar as estimativas brutas.

A `confidence` gerada pelo baseline é **heurística**, não uma probabilidade calibrada. Todos os termos que a compõem são exportados para que possamos recalibrá-la depois contra uma referência respiratória real.

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Rodar todas as coletas

Na raiz do projeto:

```bash
python analyze_acc2rr.py --data-root data --output-dir results
```

Para salvar também todos os sinais intermediários por amostra:

```bash
python analyze_acc2rr.py --data-root data --output-dir results --save-intermediate
```

Para testar apenas uma coleta:

```bash
python analyze_acc2rr.py --data-root data --recording data/em_pe/10rpm --output-dir results_single
```

## Parâmetros mais importantes

```bash
python analyze_acc2rr.py \
  --fmin 0.08 \
  --fmax 0.80 \
  --filter-order 4 \
  --hampel-window-s 0.40 \
  --hampel-sigma 4.0 \
  --window-s 30 \
  --hop-s 1 \
  --min-confidence 0.45
```

Sugestão para o primeiro benchmark: rodar também `--window-s 60` e comparar `summary.csv`, preservando a mesma banda e os demais parâmetros.

## Outputs

`results/summary.csv` contém uma linha por coleta com, entre outros:

- frequência nominal e efetiva;
- jitter, gaps e amostras ausentes estimadas;
- percentual de amostras tratadas pelo Hampel;
- magnitude e orientação da gravidade;
- variância explicada pela PC1;
- `rr_psd_raw_bpm`, `rr_psd_bpm`, `rr_acf_bpm` e `rr_final_bpm`;
- erro absoluto em relação ao valor nominal da pasta (`10rpm`/`20rpm`);
- qualidades espectral, ACF, movimento e concordância;
- confiança final;
- fração de janelas válidas;
- MAE das janelas brutas e suavizadas.

`results/report.md` resume as oito coletas em uma tabela e calcula MAE/RMSE globais.

Para cada coleta existe ainda:

```text
results/<postura>/<condição>/
├── metrics.json       # parâmetros + todas as métricas/intermediários escalares
├── windowed_rr.csv    # série temporal de RR e qualidade a cada 1 s
└── diagnostics.png    # XYZ, sinal filtrado, PC1, PSD, ACF e RR(t)
```

Com `--save-intermediate`, também é produzido `intermediate_signals.csv` com XYZ reamostrado, máscara Hampel, XYZ limpo, projeção relativa à gravidade, banda respiratória e respiratory surrogate.

## O que olhar primeiro no benchmark

1. **Erro completo**: `abs_error_bpm` deve ser pequeno nas oito condições.
2. **PSD vs ACF**: se discordarem por ~2×, suspeitar de harmônico/fundamental.
3. **PC1 variance ratio**: valores baixos indicam que a respiração não está bem descrita por uma única direção espacial naquela janela.
4. **Valid window fraction**: mede a estabilidade temporal; uma coleta pode acertar a média e ainda ser instável.
5. **Window MAE**: é mais informativo para o objetivo futuro de RR(t) do que apenas a estimativa da coleta completa.
6. **Postura**: compare sistematicamente em pé/lado/sentado/supino; o pipeline de gravidade deve reduzir a dependência da orientação física da cinta.

## Teste sintético

Há testes com sinais de 10 e 20 rpm, spikes e segundo harmônico forte:

```bash
python -m unittest discover -s tests -v
```

Os testes sintéticos verificam a lógica e evitam regressões, mas **não substituem** a validação nos `acc.csv` reais nem uma referência respiratória independente.

## Próximas comparações recomendadas

Depois deste baseline, os candidatos mais úteis para um benchmark controlado são:

- PCA 3D sem projeção pela gravidade;
- melhor eixo por SNR respiratória;
- magnitude vetorial `||a||`;
- tilt/ângulos relativos à gravidade;
- PCA recursiva/adaptativa;
- estimativa baseada em ridge de espectrograma em vez de janelas independentes;
- calibração do índice de confiança contra ground truth respiratório.
