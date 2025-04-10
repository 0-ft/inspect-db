# Inspect DB

A fast database for ingesting and querying Inspect eval runs.

## Aims
- Enable rapid querying and analysis of messages and metadata over hundreds of runs
- Facilitate rapid development of new analysis techniques
- Be lightweight, code-efficient, backend-agnostic, and easy to deploy

## Design
- Data models defined with SQLModel can be used with any SQL-compatible DBMS
- Parallelized ingestion of eval logs to extract and store log headers, samples, messages, scores, and metadata
- Data model should reflect inspect-ai classes as much as possible
  - `EvalLog` -> `DBEvalLog`
  - `EvalSample` -> `DBEvalSample`
  - `ChatMessage` -> `DBChatMessage`
- Consistent UUIDs for each entity
- `uv` dependency management
- Thorough static type checking

## Usage
- CLI for bulk ingestion of eval logs
- `EvalDB` class provides utility methods for querying data, and automatically reconstructs inspect-ai classes from the DB
  - `EvalDB.get_messages(self, sample_uuid: str, role: str | None = None) -> Sequence[ChatMessage]`
  - `EvalDB.get_sample(self, sample_uuid: str) -> EvalSample`
  - `EvalDB.get_log(self, log_uuid: str) -> EvalLog`

