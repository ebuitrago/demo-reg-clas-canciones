"""Contratos de entrada y salida del servicio (Pydantic)."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.variables import GENEROS


class Cancion(BaseModel):
    """Características conocidas antes del lanzamiento.

    Cualquier campo que no esté aquí (p. ej. reproducciones_sem1 o es_hit) se rechaza con 422:
    el contrato solo admite lo que existe en el momento de decidir.
    """
    model_config = ConfigDict(extra="forbid")

    bailabilidad: float = Field(..., ge=0, le=1, examples=[0.72])
    energia: float = Field(..., ge=0, le=1, examples=[0.65])
    valencia: float = Field(..., ge=0, le=1, examples=[0.55])
    acustica: float = Field(..., ge=0, le=1, examples=[0.12])
    tempo: float = Field(..., ge=40, le=250, description="pulsos por minuto", examples=[118.0])
    duracion_min: float = Field(..., ge=0.5, le=15, examples=[3.4])
    volumen_db: float = Field(..., ge=-60, le=5, examples=[-6.5])
    anio: int = Field(..., ge=1990, le=2035, description="año de lanzamiento", examples=[2026])
    colaboracion: int = Field(..., ge=0, le=1, description="1 si es una colaboración", examples=[1])
    genero: Literal[tuple(GENEROS)] = Field(..., examples=["urbano"])


class Prediccion(BaseModel):
    """Una predicción con la decisión que sostiene. Los campos dependen de la tarea del modelo."""
    tarea: Literal["regresion", "clasificacion"]
    modelo: str = Field(..., description="nombre y fecha del modelo cargado")
    # regresión
    popularidad_esperada: Optional[float] = None
    rango: Optional[List[float]] = Field(None, description="popularidad esperada ± MAE del modelo")
    corte_promocion: Optional[float] = None
    # clasificación
    probabilidad_hit: Optional[float] = None
    umbral: Optional[float] = None
    # decisión
    promocionar: bool
    explicacion: str


class Salud(BaseModel):
    estado: Literal["ok", "sin_modelo"]
    modelo_cargado: bool
    ruta_modelo: str
    modelo: Optional[str] = None
