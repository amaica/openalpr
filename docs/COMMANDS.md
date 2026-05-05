# Comandos executados / referência rápida

Documentação dos comandos usados no desenvolvimento e testes (OpenALPR + `mini_enhancer`).  
Caminhos assumem a raiz do repositório: `/home/aurelio/FONTES/C++/openalpr` (ajusta se necessário).

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
./build/src/alpr -c br --config config/openalpr.conf.defaults /caminho/imagem.png
```

Exemplos com imagens na raiz do projeto:

```bash
./build/src/alpr -c br --config config/openalpr.conf.defaults 1.png
./build/src/alpr -c br --config config/openalpr.conf.defaults 2.png
./build/src/alpr -c br --config config/openalpr.conf.defaults crop1.png
```

Com imagem já processada pelo `mini_enhancer`:

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
mini_enhancer --alpr foto.png
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

Saída:

```text
PLATE=IZE0J66 CONF=92.3402 USED=/.../2_enhanced_alpr_max.png
```

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
