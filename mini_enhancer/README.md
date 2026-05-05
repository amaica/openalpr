# mini_enhancer

Small standalone CLI to lightly sharpen images with OpenCV (CPU only). Not part of the OpenALPR build; lives in-repo for convenience.

## Build

```bash
cd mini_enhancer
cmake -B build -S .
cmake --build build -j2
```

Requires OpenCV development packages (`libopencv-dev` on Debian/Ubuntu).

## Run (from the project folder)

```bash
./build/mini_enhancer ../TESTECARROS.png ../mercosul.png
```

**Presets OCR / ALPR** (melhor em crop da placa; testar A/B no teu fluxo):

- `--alpr` — só **bilateral** leve (por defeito o mais seguro; evita estragar imagens já boas).
- `--alpr-max` — CLAHE + bilateral + unsharp leve (para frames muito escuros/ruidosos; pode piorar outras).

```bash
./build/mini_enhancer --alpr recorte_placa.png
./build/mini_enhancer --alpr-max foto_dificil.png
```

Saídas: `*_enhanced_alpr.ext` e `*_enhanced_alpr_max.ext`.

## Use from anywhere (PATH)

**Option A — install for your user** (binary in `~/.local/bin`):

```bash
cd mini_enhancer
cmake -B build -S .
cmake --build build -j2
cmake --install build --prefix "$HOME/.local"
```

Ensure `~/.local/bin` is on your PATH (add to `~/.bashrc` if needed):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then, in any directory:

```bash
mini_enhancer /caminho/foto.jpg
```

**Option B — symlink** (after building):

```bash
ln -sf /caminho/completo/para/openalpr/mini_enhancer/build/mini_enhancer ~/.local/bin/mini_enhancer
```

**Option C — system-wide** (needs `sudo`):

```bash
sudo cmake --install build --prefix /usr/local
```

Output files are written next to the inputs: `name_enhanced.ext`, `name_enhanced_alpr.ext`, ou `name_enhanced_alpr_max.ext`.

## Notes

- OpenCV internal threading is forced to 1 thread; a tiny pool (max 2 workers) processes multiple inputs.
- **Geral:** `blur(3×3)` → `GaussianBlur` (σ 0.8) → unsharp `addWeighted(1.2, -0.2)`.
- **`--alpr` / `--alpr-max`:** nem sempre melhoram o ALPR; comparar sempre com o original.
