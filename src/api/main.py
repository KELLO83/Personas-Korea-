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
    graph_quality,
    graph_viz,
    influence,
    insight,
    lifestyle_map,
    path,
    persona,
    recommend,
    search,
    similar,
    stats,
    target_persona,
)

logger = logging.getLogger(__name__)


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
    app.include_router(insight.router)
    app.include_router(similar.router)
    app.include_router(communities.router)
    app.include_router(path.router)
    app.include_router(persona.router)
    app.include_router(stats.router)
    app.include_router(search.router)
    app.include_router(graph_viz.router)
    app.include_router(compare.router)
    app.include_router(influence.router)
    app.include_router(recommend.router)
    app.include_router(chat.router)
    app.include_router(target_persona.router)
    app.include_router(lifestyle_map.router)
    app.include_router(career_transition.router)
    app.include_router(graph_quality.router)
    add_exception_handlers(app)
    logger.info("FastAPI application initialized")
    return app


app = create_app()
