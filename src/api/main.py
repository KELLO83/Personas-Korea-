import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..logging_config import configure_logging
from .exceptions import add_exception_handlers
from .routes import (
    chat,
    career_transition,
    communities,
    compare,
    graph_insights,
    graph_quality,
    graph_viz,
    influence,
    insight,
    lifestyle_map,
    operations,
    path,
    persona,
    rag_traces,
    recommend,
    search,
    similar,
    stats,
    target_persona,
)

logger = logging.getLogger(__name__)

API_ROUTERS = (
    insight.router,
    similar.router,
    communities.router,
    path.router,
    persona.router,
    stats.router,
    search.router,
    graph_viz.router,
    compare.router,
    influence.router,
    recommend.router,
    chat.router,
    target_persona.router,
    lifestyle_map.router,
    career_transition.router,
    graph_insights.router,
    graph_quality.router,
    operations.router,
    rag_traces.router,
)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Korean Persona KG API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in API_ROUTERS:
        app.include_router(router)
    add_exception_handlers(app)
    logger.info("FastAPI application initialized")
    return app


app = create_app()
