# Mini-GPT para Tiny Shakespeare

Implementación de un modelo GPT mínimo a nivel de carácter para los Tasks 1 y 2. El modelo usa vocabulario de 65 caracteres, atención causal multicabeza y tres bloques Transformer.

## Contenido

- `src/data.py`: descarga el corpus, crea los mapeos carácter/índice, conserva el orden temporal en la partición 90/10 y genera batches `X`, `Y` desplazados.
- `src/model.py`: Mini-GPT con `block_size=128`, `d_model=128`, `n_heads=4` y `n_layers=3`.
- `src/train.py`: entrena durante 10 épocas, muestra pérdidas y perplejidad, y guarda el checkpoint y la gráfica.
- `src/sampling.py`: implementaciones desde cero de greedy, top-k y top-p con temperatura.
- `src/generate.py`: función `generate(prompt, max_new_tokens, strategy, **kwargs)` y cargador del modelo entrenado.
- `src/generate_samples.py`: produce las cinco muestras de 200 caracteres para cada configuración del Task 2.1.
- `RESPUESTAS.md`: análisis teórico solicitado en el Task 2.2, con referencias a las muestras generadas.

## Requisitos

Python 3.10+ con PyTorch y Matplotlib:

```bash
pip install torch matplotlib
```

## Ejecución

Desde la raíz del repositorio, entrene el modelo. Si `input.txt` no existe, el script descarga el corpus oficial automáticamente.

```bash
cd src
python train.py
```

Esto crea `resultados/minigpt_weights.pth` y `resultados/loss_curve.png`. Una vez que el checkpoint existe, genere las 25 muestras del Task 2:

```bash
cd src
python generate_samples.py
```

El resultado se guarda en `resultados/task2_samples.md`. Cada muestra contiene `ROMEO:` seguido de exactamente 200 caracteres nuevos. Las semillas están fijadas por muestra para que los resultados estocásticos se puedan reproducir.

## Estrategias disponibles

| Configuración | Estrategia | Parámetros |
| --- | --- | --- |
| A | greedy | — |
| B | top-k | `k=5`, `tau=0.7` |
| C | top-k | `k=50`, `tau=1.0` |
| D | top-p | `p=0.70`, `tau=0.7` |
| E | top-p | `p=0.95`, `tau=1.0` |

`generate` conserva como contexto únicamente los últimos 128 tokens, que es el máximo admitido por el modelo. Para usarla desde otro script, cree un `TextGenerator` con `load_generator` o llame `configure_generator(model, dataset, device)` antes de invocar la función global.
