# OpenALPR (fork) — Brasil / Mercosul

Fork do OpenALPR em C++ com foco em **placas brasileiras** (Mercosul e formato antigo), **cenário de garagem** e ferramentas para integrar em produção sem reinventar o pipeline.

Isto não é um produto comercial nem um substituto oficial do OpenALPR original; é código aberto para quem precisa de ALPR no terreno, com configs e scripts pensados para o dia a dia.

---

## O que existe aqui (em termos concretos)

- **Detecção de região de placa:** o runtime usa o **detector clássico** do OpenALPR (LBP CPU por defeito; outras variantes clássicas no `openalpr.conf`).
- **OCR:** **Tesseract** no caminho principal. Se, numa região já detectada, o OCR local + pós-processamento não produzirem uma placa válida, podes activar **fallback na nuvem** via plugin **DeepSeek** (HTTP). Só corre nessa situação — não substitui o detector nem o Tesseract no dia-a-dia.
- **Cenário garagem / perfis:** `scenario = garagem` e perfis (`default`, `moto`, `garagem`) ajustam burst de OCR, votação e reforço de imagem no crop; útil para leituras repetidas ou placas difíceis.
- **Híbrido BR:** lógica extra para conciliar tentativas `eu` / `br2` / `br` e favorecer formatos de placa de carro BR válidos quando faz sentido.
- **Ferramentas do repo:**
  - `alpr` — CLI (`-c br --config … imagem`).
  - `garage_alpr.sh` — ALPR no original, `mini_enhancer --alpr-max` quando aplicável, nova leitura; saída legível ou `--json`.
  - `garage_alpr_daemon` + `scripts/garage_alprd_ctl.sh` — processo long-lived para não recarregar o motor a cada pedido.
  - `alpr-tool` — utilitário com `preview`, ROI, afinação, etc.
- **Documentação de comandos:** [docs/COMMANDS.md](docs/COMMANDS.md) (build, daemon, benchmarks, testes).

Para **só OCR** sobre um recorte que a tua aplicação já produziu, usa **`skip_detection = 1`** e regiões de interesse correctas.

---

## Fallback DeepSeek (OCR na nuvem)

Quando `ocr_fallback_enabled` está ligado e `ocr_fallback_plugin = deepseek`, o motor tenta a API DeepSeek **após** falha do OCR local numa dada candidatura de placa. É preciso **chave** e **rede**:

- **Chave (ordem):** variável de ambiente `DEEPSEEK_API_KEY`, ou `deepseek_api_key` / `deepseek_api_key_file` no `openalpr.conf` (ver comentários em `config/openalpr.conf.defaults`).
- **Conta:** criar chave em [platform.deepseek.com](https://platform.deepseek.com/); a API exige saldo/créditos activos.
- **Limitação:** o fallback actua sobre **regiões que o detector clássico já encontrou**. Se não houver nenhuma detecção, não há crop automático para enviar ao DeepSeek dentro deste fluxo.

Detalhes e exemplos: secção *Fallback DeepSeek* em [docs/COMMANDS.md](docs/COMMANDS.md).

---

## Compilar

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
```

O executável principal costuma ser `build/src/alpr`.

---

## Uso rápido (imagem)

```bash
./build/src/alpr -c br --config config/openalpr.conf.defaults caminho/imagem.png
```

Fluxo **garagem** com enhance opcional:

```bash
./garage_alpr.sh caminho/imagem.png
./garage_alpr.sh --json caminho/imagem.png
```

Mais exemplos: [docs/COMMANDS.md](docs/COMMANDS.md).

---

## `alpr-tool preview` (vídeo / linha de contagem)

```bash
./build/src/alpr-tool preview --profile=garagem --country=br --source /path/to/video.mp4
```

Detalhes: [docs/alpr_tool.md](docs/alpr_tool.md) e [docs/CONFIG_STUDY.md](docs/CONFIG_STUDY.md).

---

## Desempenho e integração

- **Perfil:** `moto` e `garagem` aumentam tentativas de OCR e votação (mais CPU, melhor em casos difíceis).
- **Resolução / ROI:** `max_detection_input_*` e ROI estáveis costumam ser o maior ganho em tempo.
- **Recorte externo:** `skip_detection` com regiões fiáveis reduz trabalho no detector clássico.
- **Métricas:** `votes_emitted`, `final_plate_count`, etc., no relatório do tool para comparar configs.

---

## Configuração

- Base: [config/openalpr.conf.defaults](config/openalpr.conf.defaults)
- Chaves: [docs/CONFIG_STUDY.md](docs/CONFIG_STUDY.md)

---

## English (short)

Brazil-focused **OpenALPR C++ fork**: classic plate detection, **Tesseract** OCR, optional **DeepSeek** cloud fallback on failed OCR (config + API key), garage scripts and daemon, BR hybrid. See [docs/COMMANDS.md](docs/COMMANDS.md).

---

## Disclaimer

Projeto open source; não é afiliação oficial com OpenALPR Inc.
