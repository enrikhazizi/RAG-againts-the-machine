from pydantic import BaseModel, Field


class Metric_tester(BaseModel):
    total_doc: int = Field(ge=0)
    total_relevant: int = Field(ge=0)

    def precision_at_k(self, relevant_count, k):
        if relevant_count < 0 or k < 0:
            raise ValueError("All input values must be non-negative.")

        if k == 0:
            return 0.0

        return relevant_count / k

    def recall_at_k(self, relevant_count):
        if relevant_count < 0:
            raise ValueError("All input values must be non-negative.")

        if self.total_relevant == 0:
            return 0.0

        return relevant_count / self.total_relevant
