# Comandos executados / referência rápida

Documentação dos comandos usados no desenvolvimento e testes (OpenALPR + `mini_enhancer`).  
Caminhos assumem a raiz do repositório: `/home/aurelio/FONTES/C++/openalpr` (ajusta se necessário).  
**Imagens de referência na raiz:** use `1.png` e `2.png` (ficheiros locais; por defeito são ignorados pelo git).

---

## OpenALPR — compilar

```bash
cd /home/aurelio/FONTES/C++/openalpr
cmake -B build -S .
cmake --build build -j4
```

---

## OpenALPR — executar `alpr` (Brasil, config padrão do repo)

```bash
cd /home/aurelio/FONTES/C++/openalpr
./build/src/alpr -c br --config config/openalpr.conf.defaults 1.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 2.png
```

Com imagem já processada pelo `mini_enhancer` (saídas derivadas de `1.png` / `2.png`):

```bash
./build/src/alpr -c br --config config/openalpr.conf.defaults 2_enhanced_alpr.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 2_enhanced_alpr_max.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 1_enhanced_alpr.png
```

**Nota:** `runtime_dir` no config deve apontar para o `runtime_data` deste clone (ex.: `runtime_dir = ./runtime_data`) e o comando costuma ser executado a partir da raiz do repo para resolver caminhos relativos.

---

## OpenALPR — debug do híbrido BR (log das tentativas `eu` / `br2` / `br`)

```bash
cd /home/aurelio/FONTES/C++/openalpr
sed 's/^debug_general.*=.*/debug_general         = 1/' config/openalpr.conf.defaults > /tmp/alpr_debug.ini
./build/src/alpr -c br --config /tmp/alpr_debug.ini 2.png
```

---

## `mini_enhancer` — compilar

```bash
cd /home/aurelio/FONTES/C++/openalpr/mini_enhancer
cmake -B build -S .
cmake --build build -j2
```

O executável gerado: `mini_enhancer/build/mini_enhancer` (target CMake interno: `enhancer`).

---

## `mini_enhancer` — uso (nitidez geral)

```bash
cd /home/aurelio/FONTES/C++/openalpr
./mini_enhancer/build/mini_enhancer /home/aurelio/FONTES/C++/openalpr/1.png
```

Saída: `*_enhanced.*` ao lado do ficheiro de entrada.

---

## `mini_enhancer` — preset para ALPR (conservador)

```bash
./mini_enhancer/build/mini_enhancer --alpr /home/aurelio/FONTES/C++/openalpr/1.png
```

Saída: `*_enhanced_alpr.*`

---

## `mini_enhancer` — preset ALPR “forte” (frames escuros/ruidosos)

```bash
./mini_enhancer/build/mini_enhancer --alpr-max /home/aurelio/FONTES/C++/openalpr/2.png
```

Saída: `*_enhanced_alpr_max.*`

---

## `mini_enhancer` — instalar no PATH do utilizador

```bash
cd /home/aurelio/FONTES/C++/openalpr/mini_enhancer
cmake -B build -S .
cmake --build build -j2
cmake --install build --prefix "$HOME/.local"
```

Garantir `export PATH="$HOME/.local/bin:$PATH"` no shell. Depois:

```bash
mini_enhancer --alpr 1.png
```

---

## Cadeia de teste (enhance + ALPR) — exemplo

```bash
cd /home/aurelio/FONTES/C++/openalpr
./mini_enhancer/build/mini_enhancer --alpr-max 2.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 2.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 2_enhanced_alpr_max.png
```

---

## Recorte de placa + comparação `full` vs `crop` (script)

Usa o JSON do `alpr` na imagem completa para obter o quadrilátero da placa, grava `artifacts/plate_crops/<N>_plate_crop.png` e volta a correr o `alpr -j` **só no recorte** (detector ainda activo). Mostra `confidence`, `processing_time_ms` por deteção e `processing_time_ms` total.

```bash
cd /home/aurelio/FONTES/C++/openalpr
python3 scripts/crop_plate_from_alpr_json.py 1.png 2.png
# opcional: tenta também --skip-detection (neste fork costuma vir vazio no crop)
python3 scripts/crop_plate_from_alpr_json.py 1.png 2.png --try-skip-detection
```

Repetir o benchmark (5 corridas por defeito; `REPEAT=10 ./scripts/benchmark_performance.sh`):

```bash
cd /home/aurelio/FONTES/C++/openalpr
./scripts/benchmark_performance.sh
```

---

## Utilitário “garagem” (auto enhance por padrão antigo)

Regra:

- Se o melhor candidato do **original** casar com **placa antiga** `@@@####` (`^[A-Z]{3}[0-9]{4}$`), **não** aplica enhance.
- Caso contrário, aplica `mini_enhancer --alpr-max` e re-roda o `alpr`.

```bash
cd /home/aurelio/FONTES/C++/openalpr
chmod +x ./garage_alpr.sh
./garage_alpr.sh 1.png
./garage_alpr.sh 2.png
```

Concorrência (câmera de entrada + saída ao mesmo tempo):

- Por padrão limita a **2 execuções em paralelo** (para não derrubar CPU fraca).
- Usa arquivos temporários em `/tmp` para evitar colisão de saída quando chegam requisições simultâneas.

```bash
MAX_PARALLEL=2 ./garage_alpr.sh 2.png
```

Saída (texto, padrão humano):

```text
PLATE=IZE0J66 CONF=92.3402 USED=/.../2_enhanced_alpr_max.png
```

Saída JSON (integração / automação): uma linha em stdout em caso de sucesso; em falha, uma linha JSON em stderr.

```bash
./garage_alpr.sh --json 2.png
```

Códigos de saída estáveis do `garage_alpr.sh`:

| Código | Significado |
|--------|-------------|
| 0 | Sucesso (placa lida) |
| 1 | Argumento em falta ou opção inválida |
| 2 | Ficheiro de imagem não encontrado |
| 3 | Falha do `mini_enhancer` |
| 4 | Sem placa após original (e enhanced, se aplicável) |
| 5 | `alpr` ou `mini_enhancer` inexistente / não executável |

### Teste automático do contrato (CI / local)

```bash
chmod +x ./scripts/test_garage_alpr_contract.sh
./scripts/test_garage_alpr_contract.sh
```

Com `ctest` (o alvo de testes fica em `build/src/tests`; use `--test-dir`):

```bash
ctest --test-dir /home/aurelio/FONTES/C++/openalpr/build/src/tests -R garage_alpr_contract -V
```

### Python no repo (sem copiar `subprocess` à mão)

O ficheiro `python/garage_alpr_client.py` chama o mesmo `garage_alpr.sh --json` (mantém enhance + regra do padrão antigo).

```bash
cd /home/aurelio/FONTES/C++/openalpr
python3 python/garage_alpr_client.py 2.png
```

Ou importando:

```python
from pathlib import Path
import sys
sys.path.insert(0, "/home/aurelio/FONTES/C++/openalpr/python")
from garage_alpr_client import read_plate

print(read_plate("2.png", repo_root=Path("/home/aurelio/FONTES/C++/openalpr")))
```

Variável de ambiente opcional: `OPENALPR_REPO` aponta para a raiz do clone (se não importares com `repo_root=`).

Teste `unittest` no repo (valida `1.png` e `2.png` quando existem na raiz):

```bash
python3 python/test_garage_alpr_client.py
```

**Nota:** compilar o projeto e usar só `openalpr.openalpr.Alpr` (binding) **não** reproduz o wrapper: é a API C++ directa. Para paridade com `./garage_alpr.sh`, usa o cliente acima ou replica tu a lógica (não recomendado).

---

## Python — chamando `garage_alpr.sh` e pegando só a placa

Exemplo simples (retorna apenas a string da placa):

```python
import subprocess

def read_plate(image_path: str) -> str:
    p = subprocess.run(
        ["./garage_alpr.sh", image_path],
        cwd="/home/aurelio/FONTES/C++/openalpr",
        capture_output=True,
        text=True,
        check=True,
    )
    # stdout: "PLATE=IZE0J66 CONF=92.3402 USED=/tmp/...\n"
    out = p.stdout.strip()
    plate = out.split("PLATE=")[1].split()[0]
    return plate

print(read_plate("2.png"))
```

Com `--json` (recomendado para parsing robusto):

```python
import json, subprocess, sys

def read_plate_json(image_path: str) -> dict:
    p = subprocess.run(
        ["./garage_alpr.sh", "--json", image_path],
        cwd="/home/aurelio/FONTES/C++/openalpr",
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        try:
            err = json.loads(p.stderr.strip() or "{}")
        except json.JSONDecodeError:
            err = {"error": p.stderr.strip() or "unknown", "code": p.returncode}
        raise RuntimeError(f"{err.get('error', 'alpr_failed')} (exit {p.returncode})")
    return json.loads(p.stdout.strip())

print(read_plate_json("2.png"))
```

Para controlar concorrência (opcional):

```python
import os, subprocess

env = {**os.environ, "MAX_PARALLEL": "2"}
subprocess.run(["./garage_alpr.sh", "2.png"], cwd="/home/aurelio/FONTES/C++/openalpr", env=env)
```

---

## Verificação rápida de ficheiros na raiz

```bash
ls -la /home/aurelio/FONTES/C++/openalpr/*.png
```

---

*Última atualização: documentação alinhada aos fluxos usados em sessão de desenvolvimento e testes A/B (`1.png`, `2.png`).*
