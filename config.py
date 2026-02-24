"""
config.py — Application settings loaded from .env
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os

class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    llama_api_key: str = ""
    llama_api_base: str = "https://openrouter.ai/api"
    llama_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()
