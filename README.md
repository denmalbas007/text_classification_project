# Проект по классификации текстов

Этот проект предназначен для классификации текстов с использованием модели машинного обучения. Он использует набор данных "20 Newsgroups" и включает скрипты для предобработки данных, тренировки модели, предсказаний и развертывания сервиса с использованием Docker и Flask.

## Установка и запуск

### Требования

- Docker
- Python 3.12
- pip

### Шаги для установки и запуска

1. **Клонирование репозитория:**

   ```bash
   git clone https://github.com/denmalbas007/text_classification_project.git
   cd text_classification_project
2. **Установка зависимостей:**

```bash
pip install -r requirements.txt
Построение Docker-образа:
```
```bash
docker build -t text_classification_service .
```
Запуск Docker-контейнера:
```bash
docker run -p 8000:8000 text_classification_service
```

3. **Использование curl**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '[
    {"text": "I love programming with Python."},
    {"text": "Machine learning is fascinating."}
]'
