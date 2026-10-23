.PHONY: install data train intervals explain rag llm llm-eval api app test all
install:   ; pip install -r requirements.txt
data:      ; python -m src.data
train:     ; python -m src.train
intervals: ; python -m src.intervals
explain:   ; python -m src.explain
rag:       ; python -m src.rag build && python -m src.rag eval
llm:       ; python -m src.llm_extract --limit 50
llm-eval:  ; python -m src.llm_extract --evaluate
api:       ; uvicorn src.api:app --reload
app:       ; streamlit run app.py
test:      ; pytest -q
all: data train intervals explain rag test
