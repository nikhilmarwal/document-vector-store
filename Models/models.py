from pydantic import BaseModel

class RewrittenQuery(BaseModel):
	initial_query: str
	new_query: str
