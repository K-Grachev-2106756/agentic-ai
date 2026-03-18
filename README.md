# RAGs and Agents

## Agents: `/src/langchain_syntax/agents/`

```
Основные технологии:
- LangChain
- MCP-servers
- Middleware
- Runtime context
- MistralAI
```

Представлены 5 реализаций агентов:

1) Базовый Агент, `base_agent.py`
    - web-search инструмент

2) MCP-Агент, `mcp_agent.py`
    - реализован MCP-сервер
    - агент инициализируется с помощью запроса к MCP-серверу

3) Runtime-Агент, `runtime_context_agent.py`
    - имеет свои state и context
    - собирает и обновляет данные о пользователе
    - учитывает неизменяемый context при ответе

4) Middleware-Агент, `middleware_agent.py`
    - модель выбирается динамически в зависимости от длины диалога (длиннее диалог - больше модель)
    - в зависимости от факта аутентификации пользователя меняются промпты и набор инструментов 
    - human in the loop для подтверждения действий
    - автоматическая оценка ответа пользователя для принятия решения Edit/Reject/Approve

5) Мульти-Агент, `multi_agent.py`
    - MCP-инструмент для поиска билетов
    - Инструменты для работы с SQL DB
    - Некоторые агенты реализованы как инструменты для главного агента

## RAGs: `/src/langchain_syntax/rag/`

```
Основные технологии:
- LangChain, LangChain цепочки
- ChromaDB
- Reranker
- MistralAI
```

Представлены 3 реализации RAG:

1) Базовый RAG

2) Self-RAG, [[arxiv.org]](https://arxiv.org/abs/2310.11511)

3) Fusion-RAG, [[arxiv.org]](https://arxiv.org/abs/2402.03367)

Выполнены с базой знаний, основанной на медицинском датасете симптомов и диагнозов `BI55/MedText`, [[huggingface.co]](https://huggingface.co/datasets/BI55/MedText)

----

Мои RAG-проекты, которые более prod-ориентированы:

1) SGR RAG для портала Госуслуг Санкт-Петербурга, [[github.com]](https://github.com/K-Grachev-2106756/rag_spb_gosuslugi)
    - without LangChain
    - Schema Guided Reasoning
    - ChromaDB
    - FastAPI
    - Docker, Docker-compose
    - pytest

2)  RAG по телеграм-каналу РБК, [[github.com]](https://github.com/K-Grachev-2106756/rag_tg_2025)
    - LangChain
    - LLM as a Judge
    - SQLAlchemy + Qdrant
    - FastAPI
    - Docker
