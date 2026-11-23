# Dashboard — Equipo 04

Página web (un solo `index.html`, sin librerías ni internet) que muestra lo que publica la API:

- Precio actual de súper, regular y diésel, con su cambio contra la semana anterior.
- Predicción de la semana siguiente con la probabilidad de que baje, se mantenga o suba.
- Gráfica del precio semanal (con tabla), filtrable por periodo y combustible.
- Exactitud del modelo elegido contra la línea base y comparación de todos los modelos por ventana de evaluación.
- Predicciones pasadas contra lo que realmente pasó.
- Estado de la API y de cada tabla Oro.

No guarda datos: todo lo pide a la API (`/health`, `/precios`, `/precios/actual`, `/prediccion`,
`/prediccion/historial`, `/modelo/metricas`).

## Usarlo en tu computadora

1. Levanta la API (ver `api/README.md`); por defecto queda en `http://localhost:8004`.
2. Sirve esta carpeta:

   ```bash
   cd dashboard
   python -m http.server 8080
   ```

3. Abre <http://localhost:8080>.

## Usarlo contra la API del Droplet

Escribe la URL en el campo **URL de la API** (por ejemplo `http://<ip>:8004`) y presiona **Conectar**,
o ábrelo con `?api=`: `http://localhost:8080/?api=http://<ip>:8004`. La última URL usada se recuerda en el navegador.

La API permite peticiones desde otros orígenes (`CORSMiddleware`), así que el dashboard puede estar en otra máquina o puerto.
