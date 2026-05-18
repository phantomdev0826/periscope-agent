from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("SEARCH_PROVIDER", "mock")
os.environ.setdefault("DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent_test")
