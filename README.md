# veiculos-tracking
The vehicle-tracking project is a facilitator in the development of the traffic counting project, its objective is to count and classify vehicles and generate reports.

# Projeto: Contador e Classificador de Veículos com DeepSORT e YOLOv8

## Requisitos
- Python 3.8+
- Instalar dependências:
  pip install -r requirements.txt

## Arquivos
- main.py: código principal com rastreamento de veículos.
- video.mp4: adicione aqui seu vídeo de teste.
- yolov8n.pt: baixe o modelo em https://github.com/ultralytics/ultralytics

## Como rodar
1. Coloque o vídeo como 'video.mp4' na pasta do projeto.
2. Baixe o modelo 'yolov8n.pt' e coloque na mesma pasta.
3. Execute:
   python main.py

Pressione 'q' para sair da janela de vídeo.
O arquivo 'detecao_veiculos.csv' será gerado com as informações de contagem e rastreamento.
