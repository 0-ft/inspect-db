from pydantic import BaseModel


class EvalSampleLocator(BaseModel):
    """A locator for a specific sample in a specific eval log."""

    location: str
    sample_id: str
    epoch: int
