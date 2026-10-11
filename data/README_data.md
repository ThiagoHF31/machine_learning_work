# Dados do projeto - Previsão do tempo em Macau
* Fonte utilizada: Open-Meteo (ERA5)
* Local: Macau (lat 22.1987, lon 113.5439), fuso Asia/Macau
* Período: 2014-01-01 a 2025-12-31 (dados diários, 4383 registros)
* Variáveis: tempmax, tempmin, temp, feelslikemax, feelslikemin, dew, humidity, precip, precipcover, windgust, windspeed, winddir, sealevelpressure, cloudcover, solarenergy, conditions
* Reproduzir: rode o notebook weather_prediction_macau.ipynb. Ele lê data/raw/*.csv se existir;
  senão baixa via Visual Crossing (export VC_API_KEY=...) ou Open-Meteo Archive API (sem chave):
  https://archive-api.open-meteo.com/v1/archive?latitude=22.1987&longitude=113.5439&start_date=2014-01-01&end_date=2025-12-31&timezone=Asia/Macau&daily=...
* Visual Crossing (manual): https://www.visualcrossing.com/weather-query-builder/ -> Macau,
  Date range (datas passadas, ~1000 dias por download), Metric, Days, CSV -> salvar em data/raw/visualcrossing/
