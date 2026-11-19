# API de precios de combustibles — Equipo 04

API en FastAPI que lee **solo la capa Oro** (tablas Delta que escribe el notebook `Gold_pronostico`) y publica
precios semanales y la predicción de la semana siguiente.

| Endpoint | Qué devuelve |
|---|---|
| `GET /health` | Estado del servicio, de cada tabla Oro y última semana con precio |
| `GET /precios/actual` | Último precio semanal por combustible, con su cambio contra la semana anterior |
| `GET /precios?combustible=diesel&desde=2026-10-01&hasta=2026-11-30&limit=100&offset=0` | Histórico semanal con filtros y paginación |
| `GET /prediccion?combustible=super` | Predicción de la semana siguiente (`sube` / `baja` / `se_mantiene`) y su probabilidad |
| `GET /modelo/metricas?escenario=ultimas_104` | Exactitud del modelo elegido y de la línea base (repetir la dirección anterior) |
| `GET /prediccion/historial?combustible=super&limit=52` | Predicciones pasadas del modelo contra la dirección real (validación temporal) |
| `GET /docs` | Documentación interactiva (Swagger) |

Ejemplo de `/prediccion?combustible=super`:

```json
{"combustible": "super", "semana": "2026-11-17", "prediccion": "sube", "probabilidad": 0.51,
 "prob_baja": 0.20, "prob_se_mantiene": 0.29, "prob_sube": 0.51,
 "semana_base": "2026-11-10", "precio_actual": 44.32, "modelo": "gb_fred", "version_modelo_uc": 1, "generado_en": "2026-11-17"}
```

### Tablas Oro que lee (desde `GOLD_URI`)

| Tabla | Endpoints |
|---|---|
| `precio_semanal` | `/precios`, `/precios/actual` |
| `pronostico_proxima_semana` | `/prediccion` |
| `metricas_modelo` | `/modelo/metricas` |
| `predicciones` | `/prediccion/historial` |

Las guarda en memoria `CACHE_SEGUNDOS` (5 min) para no consultar S3 en cada petición; si S3 falla, sirve lo último
que leyó.

**Entorno aislado: Docker.** Toda la API (Python, librerías con versión fija y código) queda dentro de una imagen,
así que corre igual en cualquier computadora y en el servidor, sin instalar nada más que Docker y sin chocar con
lo que tengan instalado otros equipos que comparten el Droplet. El contenedor corre con un usuario sin privilegios
y tiene `HEALTHCHECK` sobre `/health`.

---

## 1. Probarla en tu computadora (sin S3)

Usa los datos de `datos_ejemplo/`: **simulados**, con el mismo esquema que escribe `Gold_pronostico`
(sirven para ver la forma de las respuestas, no son precios reales). Se regeneran con
`python scripts/generar_datos_ejemplo.py`.

**Con Docker** (necesitas [Docker Desktop](https://www.docker.com/products/docker-desktop/) abierto):

```bash
cd api
cp .env.example .env          # en Windows (PowerShell):  copy .env.example .env
docker compose up -d --build
```

Abre <http://localhost:8004/docs>. `docker compose logs -f` para ver errores, `docker compose down` para apagarla.

**Sin Docker** (solo Python 3.12+):

```bash
cd api
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
GOLD_URI=./datos_ejemplo uvicorn app.main:app --port 8004   # PowerShell: $env:GOLD_URI="./datos_ejemplo"; uvicorn ...
```

**Pruebas:**

- Automáticas (desde la raíz del repo): `pip install pytest httpx && pytest tests/`
- De usuario final con curl (Git Bash, WSL, Mac o Linux): `./tests/consultas_api.sh http://localhost:8004`

---

## 2. Despliegue en el servidor

### 2.1 En Databricks (una vez): que Oro quede en el bucket que lee la API

1. Debe existir una **external location** sobre `s3://<bucket-oro>/equipo-04/` con permiso `WRITE FILES` para quien
   corre el Job.
2. En la tarea Oro del Job (notebook `Gold_pronostico`), poner el parámetro
   `ruta_oro_s3 = s3://<bucket-oro>/equipo-04/gold`.
3. Correr el Job. Al final de cada tabla debe imprimirse `copia para la API: s3://.../gold/<tabla>`.

Desde entonces, cada ejecución del Job actualiza lo que sirve la API (la API lo vuelve a leer cada 5 minutos;
no hay que tocar el servidor).

### 2.2 En el Droplet (lo hace quien tenga la llave SSH)

```bash
# 1) Entrar al servidor
ssh -i <llave> equipo-04@<ip>

# 2) Traer el código (solo la primera vez; después: git pull)
cd /home/equipo-04
git clone <url-del-repo> equipo-04-gasolina-gt
cd equipo-04-gasolina-gt/api

# 3) Configurar (el .env se crea a mano en el servidor; nunca se sube a GitHub)
cp .env.example .env
nano .env
#   PUERTO=8004
#   GOLD_URI=s3://<bucket-oro>/equipo-04/gold
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY = credenciales de SOLO LECTURA que da el profesor
#   AWS_REGION=us-east-2
chmod 600 .env

# 4) Levantar
docker compose up -d --build
docker compose ps             # debe decir "Up (healthy)" después de ~1 minuto
curl http://localhost:8004/health
```

Desde otra computadora: `curl http://<ip>:8004/health`, o `./tests/consultas_api.sh http://<ip>:8004`.

`restart: unless-stopped` mantiene la API corriendo aunque se cierre la sesión SSH o se reinicie el servidor.

**Actualizar después de cambios en el código:** `git pull && docker compose up -d --build`
(si solo cambiaron los datos en Oro no hace falta nada).

### Si algo falla

| Síntoma | Causa probable |
|---|---|
| `/health` dice `degradado` con `AccessDenied` / `403` | Credenciales del `.env` sin permiso de lectura sobre ese prefijo |
| `/health` dice `degradado` con `not found` / `No files` / `TableNotFound` | `GOLD_URI` mal escrito, o el Job no se corrió con `ruta_oro_s3` |
| `/health` dice `ok` pero `origen_datos` es `/app/datos_ejemplo` | El `.env` del servidor no tiene `GOLD_URI` apuntando a S3: está sirviendo datos simulados |
| El notebook Gold falla al copiar a S3 (`PERMISSION_DENIED`) | Falta la external location del bucket Oro o el permiso `WRITE FILES` |
| `curl` desde afuera no responde, pero desde el servidor sí | Puerto no abierto en el firewall: pedir al profesor que abra el 8004 |
| `port is already allocated` | Otro proceso usa el puerto: revisar `PUERTO` en `.env` |
| `permission denied ... docker.sock` | El usuario `equipo-04` no está en el grupo `docker`: pedírselo al profesor |
