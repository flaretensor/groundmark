from dataclasses import dataclass

MODEL = "nvidia/Nemotron-3.5-Super-VL"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
CATALOG_RETRIEVED = "2026-11-20"
CATALOG_URL = "https://tokenfactory.nebius.com/api/public/models_info"
NANOS_PER_DOLLAR = 1_000_000_000


@dataclass(frozen=True)
class Price:
    # Nanodollars per token, so reservations and settlements use exact integers.
    input: int
    cached: int
    output: int

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> int:
        cached_tokens = min(input_tokens, cached_tokens)
        return (
            (input_tokens - cached_tokens) * self.input
            + cached_tokens * self.cached
            + output_tokens * self.output
        )


PRICES = {
    MODEL: Price(input=300, cached=60, output=900),
    EMBEDDING_MODEL: Price(input=10, cached=10, output=0),
}
