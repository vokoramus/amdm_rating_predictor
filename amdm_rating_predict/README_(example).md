# python-flask-docker
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с сайта amdm.ru (песни группы ДДТ, данные об их просмотре, кол-ве звезд и пр.)

Задача: Предсказать рейтинг песни (Nзвезд/Nпросмотров * 1000) по составу аккордов песни. Регрессия

Используемые признаки:

- ~275 бинарных признаков наличия аккорда в песне

Модель: GradientBoostingRegressor

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/vokoramus/amdm_rating_predictor.git
$ cd GB_docker_flask_example
$ docker build -t fimochka/gb_docker_flask_example .
```

### Запускаем контейнер

```
docker run -d -p 8180:8180 -p 8181:8181 -v /home/yuriy/GB/4.1_ML_in_business/course_project/repo/amdm_rating_predict/app/models:/app/models amdm_rating_predictor:v0.16
```

### Переходим на localhost:8181
Вводим список аккордов в формате ['A', 'D/C', 'Hm7'] и заполняем остальные поля

