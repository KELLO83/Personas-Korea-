from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class PersonaKGException(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}


class NotFoundException(PersonaKGException):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class BadRequestException(PersonaKGException):
    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(message, status_code=400)


class ExternalServiceException(PersonaKGException):
    def __init__(self, message: str = "External service error") -> None:
        super().__init__(message, status_code=502)


class ServiceUnavailableException(PersonaKGException):
    def __init__(self, message: str = "Service unavailable") -> None:
        super().__init__(message, status_code=503)


class UnprocessableRequestException(PersonaKGException):
    def __init__(self, message: str = "Unprocessable request") -> None:
        super().__init__(message, status_code=422)


def add_exception_handlers(app: Any) -> None:
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)

    @app.exception_handler(PersonaKGException)
    async def _handle_persona_kg_exception(_request: Request, exc: PersonaKGException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, **exc.detail},
        )

    @app.exception_handler(Exception)
    async def _handle_generic_exception(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )
