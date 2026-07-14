"""
Smoke / trace harness (workstream A.1 - observability) per il RAG (pdf-researcher).

Esegue l'agente RAG REALE fuori da Streamlit:
  - fa l'ingestion del PDF di esempio (example_docs/llm_introduction.pdf),
  - manda una query document-grounded,
  - genera i trace su LangSmith (A.1, progetto 'pdf-researcher', region EU),
  - stampa la token-attribution del callback custom (TokenUsageCallbackHandler).

Streamlit e' "mockato" a livello di import: trucco LOCALE (non tocca app.py, non
va in produzione). Il RAG e' single-callback -> niente A.2.

DIFFERENZA vs lo smoke dell'Orchestrator: qui il mock di st.cache_resource DEVE
MEMOIZZARE, perche' il client Chroma in-memory e lo store BM25 globale sono cachati;
senza memoizzazione ingestion e retrieval userebbero istanze diverse -> retrieval vuoto.

Uso:  .venv/bin/python scripts/smoke_trace.py
"""
import os
import sys
import types
import uuid

# --- 0) esegui dalla root del repo (app.py usa path relativi) -----------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

# --- 1) MOCK di streamlit (deve stare PRIMA di `import app`) -------------------
class _Universal:
    """Oggetto passe-partout: callable, attr-access, context-manager, falsy."""
    def __call__(self, *a, **k): return _Universal()
    def __getattr__(self, name): return _Universal()
    def __enter__(self): return _Universal()
    def __exit__(self, *a): return False
    def __bool__(self): return False
    def __iter__(self): return iter(())

class _SessionState(dict):
    def __getattr__(self, k):
        if k in self: return self[k]
        raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]

def _cache_resource(*dargs, **dkwargs):
    """Mima st.cache_resource MA MEMOIZZANDO (per-argomenti).
    Cruciale: get_chroma_client() e get_bm25_store() devono restituire SEMPRE
    la stessa istanza tra ingestion e retrieval, come fa il vero Streamlit."""
    def _wrap(fn):
        cache = {}
        def wrapper(*a, **k):
            try:
                key = (a, tuple(sorted(k.items())))
            except TypeError:
                key = None  # argomenti non-hashable -> slot unico
            if key not in cache:
                cache[key] = fn(*a, **k)
            return cache[key]
        return wrapper
    # supporta sia  @st.cache_resource  sia  @st.cache_resource(max_entries=3)
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])
    return _wrap

class _St:
    def __init__(self):
        self.session_state = _SessionState()
        self.cache_resource = _cache_resource
        self.cache_data = _cache_resource
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Universal() for _ in range(n)]
    def tabs(self, labels, **k):
        return [_Universal() for _ in labels]
    def __getattr__(self, name):
        return _Universal()

_st = _St()
sys.modules["streamlit"] = _st
_components = types.ModuleType("streamlit.components")
_components_v1 = types.ModuleType("streamlit.components.v1")
_components_v1.html = lambda *a, **k: None
sys.modules["streamlit.components"] = _components
sys.modules["streamlit.components.v1"] = _components_v1
# helpers.get_user_info importa questi:
_rt = types.ModuleType("streamlit.runtime")
_rt.get_instance = lambda: None
sys.modules["streamlit.runtime"] = _rt
_sr = types.ModuleType("streamlit.runtime.scriptrunner")
_sr.get_script_run_ctx = lambda: None
sys.modules["streamlit.runtime.scriptrunner"] = _sr

# --- 2) importa l'app reale (load_dotenv, costruisce l'agente RAG all'import) --
import app  # noqa: E402
from async_bg import collect_events_from_agent  # noqa: E402
from helpers import TokenUsageCallbackHandler, process_agent_events  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402

def _env(name):
    v = os.environ.get(name)
    if not v:
        return "(non impostata)"
    return v if not name.endswith("KEY") else v[:12] + "…"

print("=== stato tracing ===")
for k in ("LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
    print(f"  {k} = {_env(k)}")
print(f"  session_id : {_st.session_state.get('session_id')}")
print(f"  collection : {_st.session_state.get('collection_name')}")
print()

# --- 3) INGESTION del PDF di esempio ------------------------------------------
class _UploadShim:
    """Mima un UploadedFile di Streamlit: ha .name e .read()."""
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._data = f.read()
    def read(self):
        return self._data

PDF = "example_docs/llm_introduction.pdf"
print(f"=== ingestion: {PDF} ===")
ok = app.update_vector_db([_UploadShim(PDF)])
print(f"  update_vector_db -> {ok}")
try:
    _client = app.get_chroma_client()
    _coll = _client.get_collection(_st.session_state['collection_name'])
    print(f"  chunk nella collection: {_coll.count()}")
except Exception as e:
    print(f"  (conteggio collection non disponibile: {type(e).__name__}: {e})")
print()

# --- 4) QUERY document-grounded (per far scattare il research tool) ------------
QUERY = ("According to the uploaded document, what is a large language model "
         "and how does it work? Answer using the document.")

agent = _st.session_state['agent_for_session']
token_callback = TokenUsageCallbackHandler()
thread_id = f"smoke-{uuid.uuid4()}"
config = {"configurable": {"thread_id": thread_id}, "callbacks": [token_callback]}
inputs = {"messages": [HumanMessage(content=QUERY)]}

print(f"=== QUERY: {QUERY!r} ===")
events = collect_events_from_agent(agent, inputs, config=config, timeout=240)
final_answer, trace, usage_last = process_agent_events(events)

tool_calls = [t for t in trace if t.get("type") == "tool_call"]
print(f"  thread_id  : {thread_id}")
print(f"  tool_call(s): {[t.get('tool') for t in tool_calls] or 'NESSUNO (research tool non chiamato)'}")
ans = final_answer.content if final_answer else "(nessuna risposta finale)"
print(f"  risposta   : {str(ans)[:320]}")
print("  --- token (callback custom TokenUsageCallbackHandler) ---")
print(f"    {token_callback.get_usage_dict()}")
print(f"    (parser last-interaction: {usage_last})")
print()

# --- 5) flush dei tracer (l'agente gira in un thread bg: senza flush i trace
#         potrebbero non partire prima che il processo termini) -----------------
print("=== flush tracer LangSmith ===")
wait_for_all_tracers()
print("  done. Controlla il progetto 'pdf-researcher' su LangSmith (EU).")
