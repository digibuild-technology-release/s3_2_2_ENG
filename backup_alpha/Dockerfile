# Utilizza un'immagine di base di Python
FROM python:3.9-slim

# Imposta il percorso di lavoro all'interno del contenitore
WORKDIR /app

# Copia il file dei requisiti nella directory di lavoro
COPY requirements.txt .

# Installa le dipendenze del progetto
RUN pip install --upgrade pip
RUN pip install Flask
RUN pip install Flask-API
RUN pip install -r requirements.txt

# Copia il resto dei file del progetto nella directory di lavoro
COPY . .

# Esporta la porta su cui verrà eseguita l'applicazione
EXPOSE 5000

# Avvia l'applicazione Flask
CMD ["python", "VeoliaAPIServer.py"]