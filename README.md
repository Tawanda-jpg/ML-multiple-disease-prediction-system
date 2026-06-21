# ML-multiple-disease-prediction-system
ML Project: Building a multiple disease prediction system

An end-to-end machine learning system that predicts diabetes and heart disease risk from patient health data, served through a FastAPI backend and a Streamlit web app — fully containerized with Docker.

Enter a patient's health metrics (glucose levels, blood pressure, cholesterol, etc.) and get an instant risk prediction — along with the model's confidence — for either:

- **Diabetes** (SVC pipeline)
- **Heart Disease** (Random Forest pipeline)



## Tech Stack
| Layer | Tools |
|
| **Modeling** : scikit-learn (SVC, RandomForestClassifier), pandas, numpy 
| **Backend API** : FastAPI, Uvicorn, Pydantic
| **Frontend** : Streamlit 
| **Config Management** : pydantic-settings (`.env`-driven config) 
| **Containerization** : Docker, docker-compose 
| **Serialization** : joblib 


## Key Technical Decisions

- **Group-based train/test splitting** (`GroupShuffleSplit`, keyed on a row-signature hash) instead of a plain random split, to prevent near-duplicate rows from leaking between train and test sets and inflating evaluation metrics.
- **Single source of truth for configuration.** All paths, target columns, and hyperparameters are managed through one `pydantic-settings` `Settings` class reading from `.env`, with a validator that strips stray quoting — rather than scattering config across multiple files, which is an easy trap in a project with separate training, backend, and frontend modules.
- **Relative paths over absolute paths.** Model, dataset, and log paths are stored as relative paths in `.env` so the same configuration works identically whether the app runs locally or inside a container — no path rewriting needed between environments.
- **Service-to-service networking via Docker Compose's internal DNS** (`http://backend:8000`) rather than `localhost`, since each service runs in its own container and `localhost` inside a container refers to itself, not its sibling.
- **Two Dockerfiles, one compose file** — backend and frontend are built and scaled independently rather than bundled into a single container running multiple processes.

## What I'd Add Next

- AWS deployment (ECS Fargate / App Runner) with an architecture writeup
- CI/CD via GitHub Actions for automated builds on push
- Model monitoring and basic drift detection

## Author

Built by Tawanda Ndiyamba — https://www.linkedin.com/in/tawanda-ndiyamba-25248a2b9/

GitHub:  https://github.com/Tawanda-jpg/ML-multiple-disease-prediction-system
