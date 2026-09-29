"""Generación de texto para el Mini-GPT entrenado sobre Tiny Shakespeare."""

from pathlib import Path
from typing import Any

import torch

from data import TinyShakespeareDataset
from model import MiniGPT
from sampling import sample_greedy, sample_top_k, sample_top_p


class TextGenerator:
    """Une un modelo entrenado con su vocabulario para generar caracteres."""

    def __init__(self, model: MiniGPT, dataset: TinyShakespeareDataset, device: str):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, strategy, return_trace=False, **kwargs):
        """Genera texto a partir de ``prompt`` con greedy, top-k o top-p.

        El contexto se recorta a ``block_size`` en cada paso, por lo que la
        función puede generar secuencias de cualquier longitud sin violar la
        longitud máxima que admite el modelo.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt debe ser una cadena no vacía.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens debe ser mayor o igual que cero.")
        unknown = set(prompt) - set(self.dataset.stoi)
        if unknown:
            raise ValueError(f"El prompt contiene caracteres fuera del vocabulario: {unknown}")

        idx = torch.tensor([self.dataset.encode(prompt)], dtype=torch.long, device=self.device)
        was_training = self.model.training
        self.model.eval()
        trace = []
        try:
            for _ in range(max_new_tokens):
                context = idx[:, -self.model.block_size :]
                logits = self.model(context)[:, -1, :].squeeze(0)
                if return_trace and strategy == "top_p":
                    trace.append(_top_p_candidate_count(logits, kwargs["p"], kwargs.get("tau", 1.0)))
                next_token = _sample(logits, strategy, **kwargs)
                next_idx = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                idx = torch.cat((idx, next_idx), dim=1)
        finally:
            self.model.train(was_training)

        text = self.dataset.decode(idx.squeeze(0).tolist())
        return (text, trace) if return_trace else text


def _sample(logits, strategy: str, **kwargs: Any) -> int:
    """Despacha una estrategia y valida sus parámetros."""
    if strategy == "greedy":
        if kwargs:
            raise ValueError("greedy no acepta parámetros adicionales.")
        return sample_greedy(logits)

    tau = kwargs.pop("tau", 1.0)
    if tau <= 0:
        raise ValueError("tau debe ser mayor que cero.")
    if strategy == "top_k":
        k = kwargs.pop("k", None)
        if kwargs or not isinstance(k, int) or not 1 <= k <= logits.numel():
            raise ValueError("top_k requiere un entero k entre 1 y el tamaño del vocabulario.")
        return sample_top_k(logits, k=k, tau=tau)
    if strategy == "top_p":
        p = kwargs.pop("p", None)
        if kwargs or not isinstance(p, (int, float)) or not 0 < p <= 1:
            raise ValueError("top_p requiere p en el intervalo (0, 1].")
        return sample_top_p(logits, p=p, tau=tau)
    raise ValueError("strategy debe ser 'greedy', 'top_k' o 'top_p'.")


def _top_p_candidate_count(logits, p: float, tau: float) -> int:
    """Cantidad de caracteres que el núcleo top-p conserva en este paso."""
    probabilities = torch.softmax(logits / tau, dim=-1)
    cumulative = torch.cumsum(torch.sort(probabilities, descending=True).values, dim=-1)
    return int(torch.searchsorted(cumulative, torch.tensor(p, device=logits.device)).item() + 1)


_default_generator: TextGenerator | None = None


def configure_generator(model: MiniGPT, dataset: TinyShakespeareDataset, device: str) -> None:
    """Configura el generador usado por la función pública ``generate``."""
    global _default_generator
    _default_generator = TextGenerator(model, dataset, device)


def generate(prompt, max_new_tokens, strategy, **kwargs):
    """Genera texto usando el modelo configurado con ``configure_generator``."""
    if _default_generator is None:
        raise RuntimeError("Primero llame configure_generator(model, dataset, device).")
    return _default_generator.generate(prompt, max_new_tokens, strategy, **kwargs)


def load_generator(weights_path: str | Path, data_dir: str | Path = "..") -> TextGenerator:
    """Carga los pesos del Task 1 y devuelve un generador listo para usar."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TinyShakespeareDataset(data_dir=str(data_dir), block_size=128)
    model = MiniGPT(dataset.vocab_size, d_model=128, n_heads=4, n_layers=3, block_size=128)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return TextGenerator(model, dataset, device)
