# ASTRA — Satellite Conjunction Risk System

ASTRA is a machine-learning early-warning and risk-assessment system for
satellite conjunctions. It combines a FastAPI backend with a React-based
mission-control frontend and a trained CatBoost regression model.

## Project Structure

```text
astra-full/
├── backend/
│   ├── main.py
│   ├── predict.py
│   ├── requirements.txt
│   └── model/
│       ├── catboost_risk_model.cbm
│       └── selected_features.json
│
└── frontend/
    ├── src/
    │   └── app/
    │       ├── components/
    │       ├── pages/
    │       └── services/
    ├── package.json
    └── vite.config.ts
```

The actual folder name on an individual computer may be different. The
repository does not depend on a specific local folder name.

## System Overview

ASTRA processes structured Conjunction Data Message (CDM) data and uses a
trained `CatBoostRegressor` to estimate continuous satellite conjunction risk.

The main prediction flow is:

```text
CSV / XLSX Dataset
        ↓
Frontend Upload
        ↓
FastAPI Backend
        ↓
Data Validation & Preprocessing
        ↓
CatBoostRegressor
        ↓
Predicted log10 Collision Risk
        ↓
Risk Classification
        ↓
Low / Medium / High
```

The frontend communicates with the backend through the prediction API.

## Machine-Learning Model

The deployed model is:

```text
CatBoostRegressor
```

The model is stored in CatBoost's native format:

```text
backend/model/catboost_risk_model.cbm
```

The expected feature schema is stored in:

```text
backend/model/selected_features.json
```

The deployed model schema contains **102 leak-free features** (excluding target leakage indicators `max_risk_estimate` and `max_risk_scaling`).

The categorical feature is:

```text
c_object_type
```
FROM redis:latest
CMD ["redis-server"]
docker build -t redisnew .
docker run --name myredisnew -d redisnew
docker ps
docker stop myredisnew
docker login
docker commit myredisnew yourusername/redis1
docker push yourusername/redis1
docker rm myredisnew
docker rmi redisnew
docker pull yourusername/redis1
docker run --name myredis -d yourusername/redis1
docker exec -it myredis redis-cli
CatBoost handles this categorical feature natively.

The feature schema includes raw astrodynamics parameters, orbital element covariances, relative velocities, miss distances, time-to-TCA, and four numeric missingness indicators:

```text
covariance_missing
t_rcs_missing
c_rcs_missing
space_weather_missing
```
version: "3.8"

services:
  wordpress:
    image: wordpress:latest
    ports:
      - "8081:80"
    environment:
      WORDPRESS_DB_HOST: db:3306
      WORDPRESS_DB_USER: wordpress
      WORDPRESS_DB_PASSWORD: wordpress
      WORDPRESS_DB_NAME: wordpress
    depends_on:
      - db

  db:
    image: mysql:5.7
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: wordpress
      MYSQL_USER: wordpress
      MYSQL_PASSWORD: wordpress



      version: "3.9"

services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: demo
      POSTGRES_PASSWORD: demo
      POSTGRES_DB: demo_db
When missing from incoming raw CDMs, the backend passes them as numeric `NaN`, which CatBoost handles natively without imputation artifacts.
