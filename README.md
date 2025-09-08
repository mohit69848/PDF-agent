python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill .env with keys
streamlit run app.py # or python run_local.py