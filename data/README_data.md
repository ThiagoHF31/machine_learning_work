# Dados do projeto – Previsão do tempo em Macau

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `macao_teacher_data.csv` | Visual Crossing Weather Query Builder (download pedido pela professora) | 15 dias de **previsão** (11/10/2026 a 25/10/2026), unidades americanas. Insuficiente para treinar. |
| `macao_openmeteo.csv` | Open-Meteo Historical Weather API (reanálise ERA5) | Histórico diário bruto de Macau, 01/01/2014 a 31/12/2025 (4383 dias), nomes de colunas originais da API. |
| `processed/macau_features_processed.csv` | gerado pelo notebook | Matriz de features + alvos usada nos modelos. |

* Local: Macau (lat 22.1987, lon 113.5439), fuso Asia/Macau.
* Como reproduzir: rode `notebooks/weather_prediction_macau.ipynb`. Se `macao_openmeteo.csv` não existir, o notebook baixa de
  https://archive-api.open-meteo.com/v1/archive (sem cadastro) com as variáveis listadas em `notebooks/weather_utils.py`.
* Para conferir no navegador (10 primeiros dias, CSV):
  https://archive-api.open-meteo.com/v1/archive?latitude=22.1987&longitude=113.5439&start_date=2014-01-01&end_date=2014-01-10&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,cloud_cover_mean&timezone=Asia%2FMacau&format=csv
