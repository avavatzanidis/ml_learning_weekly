financial-ml-system/
├── src/
│   ├── api/                      # FastAPI service logic
│   ├── analytics/                # SHAP, risk analyzers, explainability
│   ├── backtesting/             # Strategy and signal backtesting engine
│   ├── config/                   # Configuration files & constants
│   ├── dashboard/                # Streamlit or web dashboard frontend
│   ├── data/                     # Data ingestion, streaming, storage
│   ├── deploy/                   # Docker, Kubernetes, CI/CD
│   ├── features/                 # Feature engineering (historical & real-time)
│   ├── models/                   # ML models, signal generation
│   ├── signals/                  # Signal orchestration, combination logic
│   ├── trading/                  # Paper trading, portfolio mgmt
│   ├── advanced/                 # News sentiment, multi-timeframe logic
│   └── utils/                    # Shared utilities & helpers
│
├── notebooks/                    # Jupyter notebooks for EDA, prototyping
├── tests/                        # Unit + integration tests
├── config/                       # Central YAML/JSON configs
├── deploy/                       # Dockerfiles, compose, Helm charts
├── docs/                         # Documentation, diagrams
│
├── .env                          # Environment variables
├── .gitignore
├── README.md
├── requirements.txt
├── docker-compose.yml
└── Makefile                      # CLI for tasks like `make test`, `make run`
