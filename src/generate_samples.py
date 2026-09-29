"""Produce las 25 muestras exigidas por el Task 2.1."""

from pathlib import Path
import argparse
import random

import torch

from generate import load_generator


CONFIGURATIONS = {
    "A": ("greedy", {}),
    "B": ("top_k", {"k": 5, "tau": 0.7}),
    "C": ("top_k", {"k": 50, "tau": 1.0}),
    "D": ("top_p", {"p": 0.70, "tau": 0.7}),
    "E": ("top_p", {"p": 0.95, "tau": 1.0}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="../resultados/minigpt_weights.pth")
    parser.add_argument("--output", default="../resultados/task2_samples.md")
    args = parser.parse_args()

    generator = load_generator(args.weights, data_dir="..")
    lines = ["# Muestras — Task 2.1", "", "Prompt: `ROMEO:`. Cada salida contiene el prompt y 200 caracteres nuevos."]
    for config_id, (strategy, kwargs) in CONFIGURATIONS.items():
        lines.extend(["", f"## Configuración {config_id}", f"Estrategia: `{strategy}`; parámetros: `{kwargs}`."])
        for sample_number in range(1, 6):
            # Semillas fijas: las configuraciones estocásticas se pueden revisar y repetir.
            seed = 2026 + ord(config_id) * 10 + sample_number
            random.seed(seed)
            torch.manual_seed(seed)
            wants_trace = config_id == "E"
            result = generator.generate("ROMEO:", 200, strategy, return_trace=wants_trace, **kwargs)
            text, trace = result if wants_trace else (result, None)
            lines.extend(["", f"### Muestra {config_id}{sample_number}", "```text", text, "```"])
            if trace is not None:
                lowest = sorted(range(len(trace)), key=trace.__getitem__)[:2]
                highest = sorted(range(len(trace)), key=trace.__getitem__, reverse=True)[:2]
                describe = lambda positions: ", ".join(
                    f"paso {pos + 1} ('{text[len('ROMEO:') + pos]}'): {trace[pos]} candidatos"
                    for pos in positions
                )
                lines.append(f"Traza top-p — menor núcleo: {describe(lowest)}; mayor núcleo: {describe(highest)}.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Se guardaron 25 muestras en {output}")


if __name__ == "__main__":
    main()
