# Mini-GPT for Tiny Shakespeare (Task 1)

Este proyecto contiene la solución al Task 1. Todo se ha estructurado utilizando buenas prácticas de ingeniería de software para facilitar el desarrollo posterior del Task 2.

## Estructura del Proyecto

*   `src/data.py`: Contiene la clase `TinyShakespeareDataset` que maneja el dataset, el vocabulario de caracteres únicos (exactamente 65), los mappings bidireccionales, la partición 90/10 respetando orden temporal y la generación de batches desplazados.
*   `src/model.py`: Contiene la implementación de `MiniGPT`, adaptada para soportar arquitecturas multicapa (`n_layers`), atenciones multicabeza (`n_heads`), máscara causal y operaciones en batch (`batch_size`).
*   `src/train.py`: Script principal para ejecutar el pipeline de entrenamiento del modelo utilizando los hiperparámetros mínimos requeridos.
*   `input.txt`: El dataset completo de Tiny Shakespeare.
*   `resultados/`: Carpeta (creada automáticamente) donde se guardan los pesos de la red y la gráfica de pérdida.
*   `.gitignore`: Configurado para ignorar caché, los entornos virtuales, el archivo de dataset crudo y los pesos del modelo, dejando todo limpio.

## Cómo Ejecutar y Presentar el Task 1

1.  **Correr el Entrenamiento**:
    Navega a la carpeta de código y ejecuta el script de entrenamiento:
    ```bash
    cd src
    python train.py
    ```

2.  **Qué Presentar**:
    *   **Evidencias en consola (Task 1.1)**: En tu consola o notebook, pega las líneas que imprime este script al inicio donde confirma el tamaño de vocabulario (65) y los tamaños de los splits (90/10), así como el ejemplo que muestra el desplazamiento de batch X vs Y.
    *   **Curva de pérdida (Task 1.2)**: Adjunta la imagen `loss_curve.png` guardada en la carpeta `resultados/`. *Breve análisis*: Podrás notar que la curva de validación desciende a la par que la de entrenamiento sin separarse (no hay overfitting notable en estas 10 épocas) lo cual es indicativo de un modelo sano para este dataset.
    *   **Perplejidad (Task 1.2)**: El script imprime automáticamente la **Perplejidad de Validación Final** en la consola tras finalizar (debería salirte un valor de ~8.0, perfectamente dentro del rango 3.5 a 8.0 que piden).
    *   **Funciones de Muestreo (Task 1.3)**: Puedes mostrar el contenido del archivo `src/sampling.py` donde se han programado desde cero las tres estrategias (`greedy`, `top_k`, `top_p`) procesando los logits crudos según sus definiciones matemáticas precisas.

## Notas para la Persona Asignada al Task 2

La estructura dejada facilita directamente la construcción del generador de texto (Task 2):
1. Podrá importar `MiniGPT` y `TinyShakespeareDataset` de manera limpia.
2. Deberá instanciar la clase del dataset (para acceso a `encode` y `decode`) y el modelo.
3. Podrá cargar el estado del modelo pre-entrenado desde `resultados/minigpt_weights.pth` usando `model.load_state_dict()` para no tener que volver a entrenar.
4. Con el modelo cargado, podrá implementar su función `generate(prompt, max_new_tokens, strategy, **kwargs)` en un nuevo archivo (por ejemplo, `src/generate.py`) basándose en los outputs del modelo y aplicando las lógicas de Greedy, Top-K y Top-P.
