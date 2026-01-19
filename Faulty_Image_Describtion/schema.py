from typing import List

from pydantic import BaseModel, ValidationError


class OutputSchema(BaseModel):
    description: str


def validate_output(payload: dict) -> List[str]:
    try:
        OutputSchema.model_validate(payload)
        return []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "validation error")
            errors.append(f"{loc}: {msg}".strip())
        return errors
