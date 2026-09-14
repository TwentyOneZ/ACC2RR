# ACC2RR

Estimador de **frequência respiratória (RR, em respirações por minuto)** a partir do acelerômetro triaxial de um sensor torácico, desenvolvido e testado inicialmente com dados do **Polar H10**.

O algoritmo processa `x/y/z` do acelerômetro, isola a faixa respiratória, procura uma representação espacial robusta do movimento torácico e estima a periodicidade respiratória combinando análise espectral e autocorrelação. Quando a primeira componente principal (PC1) se torna ambígua — por exemplo, quando um harmônico passa a dominar — o sistema pode trocar automaticamente para uma direção otimizada no plano PC1–PC2. A série final é suavizada por um filtro de Kalman ponderado pela confiança de cada observação.

> **Saída operacional recomendada:** `rr_v8_hard_kalman_bpm` em `windowed_rr.csv`.

## Como funciona

Fluxo resumido:

```text
ACC XYZ
  ↓
QC de timestamps + reamostragem uniforme
  ↓
remoção conservadora de spikes (Hampel)
  ↓
estimativa da gravidade
  ↓
projeção do movimento no plano perpendicular à gravidade
  ↓
filtro Butterworth na banda respiratória
  ↓
janelas móveis de 30 s / passo de 1 s
  ↓
PCA → PC1 e PC2
  ↓
┌──────────────────────┬────────────────────────┐
│ surrogate clássico   │ surrogate adaptativo   │
│ PC1                  │ melhor direção PC1–PC2 │
└──────────┬───────────┴───────────┬────────────┘
           ↓                       ↓
        Welch PSD + autocorrelação + consenso harmônico
                       ↓
               score de ambiguidade
                       ↓
             hard adaptive gate
                (Schmitt trigger)
                       ↓
              filtro de Kalman
                       ↓
               RR final [rpm]
```

### Técnicas principais

- **Timestamp QC:** detecta duplicatas, jitter, gaps e frequência efetiva; gaps longos não são interpolados.
- **Hampel vetorial:** remove spikes com estatística robusta; se qualquer eixo for outlier, o vetor XYZ inteiro é interpolado para preservar a geometria.
- **Referência pela gravidade:** estima o vetor gravitacional mediano e trabalha com o movimento no plano perpendicular a ele.
- **Butterworth 0,08–0,80 Hz:** isola aproximadamente 4,8–48 rpm com filtragem zero-phase (`sosfiltfilt`).
- **PCA:** identifica as direções dominantes do movimento torácico.
- **Busca PC1–PC2:** testa 36 direções (0° a 175°, passo de 5°) e escolhe a que possui melhor qualidade respiratória interna.
- **Welch PSD:** estima as frequências com maior energia.
- **Autocorrelação (ACF):** estima o período de repetição temporal do sinal.
- **Consenso harmônico:** combina PSD e ACF e considera relações de 2×/3× para reduzir confusão entre frequência fundamental e harmônicos.
- **Score de ambiguidade:** avalia quando PC1 deixa de ser uma representação confiável da respiração.
- **Schmitt trigger:** troca entre PC1 e a direção otimizada usando histerese temporal, evitando flapping.
- **Filtro de Kalman:** suaviza o RR e dá mais peso às observações de maior confiança.

A seleção do algoritmo **não usa postura, nome da condição ou um RR esperado**. Quando um `target_rpm` está disponível nos dados experimentais, ele é usado apenas posteriormente para calcular métricas de erro.

## Requisitos

- Python 3.10+ recomendado
- dependências de `requirements.txt`

Crie e ative um ambiente virtual:

### Windows / PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Formato de entrada

A entrada principal é um `acc.csv` com pelo menos estas colunas:

```text
timestamp_epoch_s,x_mg,y_mg,z_mg
```

Exemplo:

```csv
timestamp_epoch_s,x_mg,y_mg,z_mg
1726315200.000,-118,231,962
1726315200.020,-119,229,963
1726315200.040,-117,232,961
```

Um `metadata.json` na mesma pasta é opcional. Quando presente, pode informar a frequência nominal do acelerômetro, por exemplo:

```json
{
  "acc_sample_rate_hz": 50
}
```

## Executar uma gravação

Na raiz do repositório:

```powershell
.venv\Scripts\python.exe analyze_acc2rr.py --recording caminho\para\acc.csv --output-dir resultado
```

Também é possível passar a pasta que contém `acc.csv`:

```powershell
.venv\Scripts\python.exe analyze_acc2rr.py --recording caminho\para\gravacao --output-dir resultado
```

Para salvar também todos os sinais intermediários do pré-processamento:

```powershell
.venv\Scripts\python.exe analyze_acc2rr.py --recording caminho\para\gravacao --output-dir resultado --save-intermediate
```

Em Linux/macOS, use `python` ou `.venv/bin/python` no lugar de `.venv\Scripts\python.exe`.

## Processar várias gravações

O CLI procura recursivamente todos os arquivos `**/acc.csv` abaixo de `--data-root`:

```powershell
.venv\Scripts\python.exe analyze_acc2rr.py --data-root data --output-dir results
```

Exemplo de estrutura:

```text
data/
├── pessoa_01/
│   ├── teste_01/
│   │   ├── acc.csv
│   │   └── metadata.json
│   └── teste_02/
│       └── acc.csv
└── pessoa_02/
    └── teste_01/
        └── acc.csv
```

## Saídas

Para cada gravação o pipeline gera, entre outros:

```text
<output>/...
├── metrics.json
├── windowed_rr.csv
├── diagnostics.png
├── adaptive_v8.csv
└── adaptive_v8.png
```

Com `--save-intermediate`, também é criado:

```text
intermediate_signals.csv
```

No diretório raiz de saída também são gerados:

```text
summary.csv
report.md
config.json
```

### RR recomendado

A principal série operacional é:

```text
rr_v8_hard_kalman_bpm
```

em `windowed_rr.csv`.

Ela representa a estimativa de RR após:

```text
seleção adaptativa PC1 / optimized-PC12
                  ↓
              Kalman
                  ↓
            RR final [rpm]
```

Outras colunas úteis:

| Coluna | Significado |
|---|---|
| `time_s` | tempo central da janela |
| `rr_v8_hard_bpm` | observação de RR antes do Kalman |
| `rr_v8_hard_kalman_bpm` | **RR final recomendado** |
| `v8_hard_mode` | fonte selecionada (`classic_v4` ou `adaptive_opt_pc12`) |
| `v8_hard_adaptive_active` | indica se a projeção adaptativa está ativa |
| `v8_hard_selected_confidence` | confiança interna da observação selecionada |

A configuração padrão utiliza janelas de 30 s com avanço de 1 s; portanto, após haver sinal suficiente, existe aproximadamente uma nova estimativa por segundo.

## Exemplo: obter o RR mais recente

```python
import pandas as pd

rr = pd.read_csv("resultado/windowed_rr.csv")
valid = rr.dropna(subset=["rr_v8_hard_kalman_bpm"])

latest = valid.iloc[-1]

print(f"RR: {latest['rr_v8_hard_kalman_bpm']:.2f} rpm")
print(f"Confiança: {latest['v8_hard_selected_confidence']:.2f}")
print(f"Modo: {latest['v8_hard_mode']}")
```

Saída típica:

```text
RR: 18.74 rpm
Confiança: 0.81
Modo: adaptive_opt_pc12
```

## Parâmetros do CLI

Os principais parâmetros podem ser consultados com:

```bash
python analyze_acc2rr.py --help
```

Defaults importantes:

| Parâmetro | Default |
|---|---:|
| `--fmin` | 0.08 Hz |
| `--fmax` | 0.80 Hz |
| `--filter-order` | 4 |
| `--hampel-window-s` | 0.40 s |
| `--hampel-sigma` | 4.0 |
| `--window-s` | 30 s |
| `--hop-s` | 1 s |
| `--min-confidence` | 0.45 |

Os parâmetros adaptativos internos foram validados com a configuração atual e não devem ser reajustados com base apenas no dataset de desenvolvimento.

## Testes

Execute toda a suíte com:

```bash
python -m pytest -v
```

A suíte inclui testes sintéticos e regressões para processamento temporal, resolução harmônica, busca espacial, gate adaptativo e fusão de observações.

## Observações e limitações

- O algoritmo atual é adequado principalmente para **análise offline** porque utiliza filtragem zero-phase (`sosfiltfilt`).
- A janela de 30 s favorece estabilidade, mas limita a resposta a mudanças respiratórias muito rápidas.
- Movimentos corporais intensos podem contaminar a banda respiratória.
- A confiança é um índice interno heurístico, não uma probabilidade clínica calibrada.
- O método ainda deve ser validado em maior variedade de participantes, frequências respiratórias, movimento e com referência respiratória independente.
- O projeto não deve ser tratado como dispositivo médico validado.

## Estado atual

O **hard adaptive gate + optimized PC1–PC2 + Kalman** é o pipeline operacional recomendado para leitura da frequência respiratória. As demais séries exportadas permanecem disponíveis para diagnóstico, comparação e pesquisa.
