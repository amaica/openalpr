# Estudo de configurações — OpenALPR

Este documento mapeia **todas as possibilidades de configuração** do projeto: arquivo `.conf`, opções de CLI do `alpr-tool` e parâmetros do core (config.h / config.cpp).

---

## 1. Fontes de configuração

| Fonte | Uso |
|-------|-----|
| **Arquivo .conf** | `config/openalpr.conf.defaults`, `--conf <path>`, ou variável `OPENALPR_CONFIG_FILE` |
| **Variável de ambiente** | `OPENALPR_CONFIG_FILE`, `OPENALPR_RUNTIME_DATA` |
| **CLI (alpr-tool preview)** | Argumentos `--country`, `--line`, `--plates-only-past-line`, etc. (sobrescrevem/estendem o .conf) |
| **Runtime por país** | `runtime_data/config/<country>.conf` (ex.: br2.conf, br.conf) — valores por região |

---

## 2. Chaves do arquivo .conf (core)

Carregadas em `config.cpp` (loadCommonValues + loadCountryValues). Formato INI sem seção (section "").

### 2.1 Runtime e região

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `runtime_dir` | string | `./runtime_data` | Diretório de runtime_data (cascades, tessdata, configs por país) |
| `country` | string | — | País/região (ex.: br, br2, eu, us). Carregado via API, não do .conf principal |

### 2.2 Detector

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `detector_type` | string | `auto` | `auto` \| `yolo` \| `classic` |
| `detector` | string | `lbpcpu` | `lbpcpu` \| `lbpgpu` \| `lbpopencl` \| `morphcpu` |
| `yolo_model_path` | string | "" | Caminho do modelo YOLO |
| `yolo_input_width` | int | 640 | Largura de entrada YOLO |
| `yolo_input_height` | int | 640 | Altura de entrada YOLO |
| `yolo_conf_threshold` | float | 0.25 | Confiança mínima YOLO |
| `yolo_nms_threshold` | float | 0.45 | NMS threshold YOLO |
| `yolo_min_detections` | int | 1 | Mínimo de detecções |
| `detector_fallback_classic` | bool | 1 | Fallback para detector clássico |
| `detection_iteration_increase` | float | 1.1 | Aumento por iteração LBP |
| `detection_strictness` | int | 3 | Rigor da detecção (2–9) |
| `max_plate_width_percent` | float | 100 | Máx. largura da placa (%) |
| `max_plate_height_percent` | float | 100 | Máx. altura da placa (%) |
| `max_detection_input_width` | int | 1280 | Redimensionar entrada (px) |
| `max_detection_input_height` | int | 720/768 | Redimensionar entrada (px) |
| `contrast_detection_threshold` | float | 0.3 | Limiar de contraste |
| `skip_detection` | bool | 0 | 1 = pular detecção (usar ROIs/bbox externos) |
| `detection_mask_image` | string | "" | Máscara de detecção (imagem) |
| `analysis_count` | int | 1 | Número de análises por imagem |
| `prewarp` | string | "" | Parâmetros de prewarp (calibração) |
| `max_plate_angle_degrees` | int | 15 | Ângulo máximo da placa |

### 2.3 Veículo / cenário / OCR (core)

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `vehicle` | string | `car` | `car` \| `moto` |
| `scenario` | string | `default` | `default` \| `garagem` |
| `ocr_burst_frames` | int | 1 (car) / 6 (moto) / 10 (garagem) | Frames de burst OCR |
| `vote_window` | int | = ocr_burst_frames | Janela de voto temporal |
| `min_votes` | int | 1 (car) / 3 (moto, garagem) | Mínimo de votos |
| `fallback_ocr_enabled` | bool | 0 (car) / 1 (garagem) | Fallback OCR |
| `moto_upsample` | bool | 0 | Upsample para moto |
| `moto_upsample_scale` | float | 2.0 | Escala de upsample |
| `profile` | string | derivado | `default` \| `moto` \| `garagem` (derivado de vehicle+scenario) |

### 2.4 Brasil / híbrido

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `br_enable_hybrid` | bool | 1 | Pipeline híbrido br2/eu/br |
| `br_hybrid_order` | string | `br2,br` | Ordem de regiões |
| `br_hybrid_fallback_region` | string | `eu:ad` | Fallback (ex.: eu:ad) |
| `br_hybrid_min_confidence` | float | 70/80 | Confiança mínima |

### 2.5 Perfil de veículo (moto vs carro)

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `vehicle_profile_mode` | string | `auto` | `auto` \| `car` \| `moto` |
| `moto_aspect_ratio_min` | float | 0.9 | Aspect ratio mínimo para moto |
| `moto_aspect_ratio_max` | float | 2.2 | Aspect ratio máximo para moto |

### 2.6 ROI (região de interesse)

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `enable_roi` | bool | 0 | Habilitar ROI |
| `roi_x` | float | 0 | X normalizado (0–1) |
| `roi_y` | float | 0 | Y normalizado |
| `roi_width` | float | 1 | Largura normalizada |
| `roi_height` | float | 1 | Altura normalizada |

### 2.7 Pré-processamento

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `preproc_enable` | bool | 0 | Habilitar pré-processamento |
| `preproc_brightness` | float | 0 | Brilho |
| `preproc_contrast` | float | 1 | Contraste |
| `preproc_gamma` | float | 1 | Gama |
| `preproc_clahe_enable` | bool | 0 | CLAHE |
| `preproc_clahe_clip` | float | 2 | Clip CLAHE |
| `preproc_sharpen` | float | 0 | Nitidez |
| `preproc_denoise` | float | 0 | Denoise |
| `preproc_apply_before_detector` | bool | 0 | Aplicar antes do detector |

### 2.8 OCR / plugins (parser)

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `ocr_primary` | string | openalpr | OCR primário |
| `ocr_policy` | string | primary_only | primary_only \| fallback_on_low_confidence \| ensemble |
| `ocr_min_confidence` | float | 0 | Confiança mínima |
| `ocr_fallback_enabled` | bool | 0 | Fallback OCR |
| `ocr_fallback_plugin` | string | deepseek | Plugin de fallback |
| `ocr_fallback_min_confidence` | float | 80 | Confiança mínima fallback |
| `ocr_fallback_timeout_ms` | int | 800 | Timeout fallback |
| `plugins_enabled` | bool | 0 | Plugins |
| `plugins_path` | string | /opt/alpr/plugins | Caminho plugins |
| `vehicle_attrs_enabled` | bool | 0 | Atributos de veículo |
| `vehicle_attrs_plugin` | string | onnx_vehicle | Plugin |
| `vehicle_attrs_min_confidence` | float | 0.7 | Confiança |

### 2.9 Pós-processamento / regex

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `must_match_pattern` | bool | 0 | Exigir padrão |
| `postprocess_min_confidence` | float | 65/100 | Confiança mínima pós-OCR |
| `postprocess_confidence_skip_level` | float | 80 | Skip level |
| `ocr_img_size_percent` | float | 1.33 | Tamanho da imagem OCR |
| `state_id_img_size_percent` | float | 2.0 | Tamanho state ID |
| `ocr_min_font_point` | int | 6/100 | Tamanho mínimo de fonte |

### 2.10 Debug

| Chave | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `debug_general` | bool | 0 | Debug geral |
| `debug_timing` | bool | 0 | Tempos |
| `debug_prewarp` | bool | 0 | Prewarp |
| `debug_detector` | bool | 0 | Detector |
| `debug_state_id` | bool | 0 | State ID |
| `debug_plate_lines` | bool | 0 | Linhas da placa |
| `debug_plate_corners` | bool | 0 | Cantos |
| `debug_char_segment` | bool | 0 | Segmentação de caracteres |
| `debug_char_analysis` | bool | 0 | Análise de caracteres |
| `debug_color_filter` | bool | 0 | Filtro de cor |
| `debug_ocr` | bool | 0 | OCR |
| `debug_postprocess` | bool | 0 | Pós-processamento |
| `debug_show_images` | bool | 0 | Mostrar imagens |
| `debug_pause_on_frame` | bool | 0 | Pausar por frame |

### 2.11 Por país (runtime_data/config/<country>.conf)

Ex.: `br2.conf`, `br.conf` — chaves por região (plate dimensions, regex, detector_file, ocr_language, etc.):

- `min_plate_size_width_px`, `min_plate_size_height_px`
- `multiline`, `invert` (auto \| always \| never)
- `plate_width_mm`, `plate_height_mm`
- `char_height_mm`, `char_width_mm`
- `char_whitespace_*`, `template_max_*_px`
- `char_analysis_*`, `segmentation_*`
- `plateline_sensitivity_*`
- `detector_file`, `ocr_language`
- `postprocess_regex_letters`, `postprocess_regex_numbers`
- `postprocess_min_characters`, `postprocess_max_characters`

---

## 3. Uso no alpr_tool (cfg.get)

Lidas no tool a partir do `Config` já carregado (cfg.get("key", "default")):

- `video_source` — fonte de vídeo
- `enable_roi`, `roi_x`, `roi_y`, `roi_width`, `roi_height`
- `prewarp_enabled`, prewarp keys
- `speed_enabled`, `speed_mode`, `speed_line_a_y_percent`, `speed_line_b_y_percent`, `speed_dist_m`, `speed_time_source`, `speed_min_kmh`, `speed_max_kmh`, `speed_smoothing`, `speed_ema_alpha`, `speed_log`, `speed_require_plate`
- `country`, `runtime_dir`, `skip_detection`
- `vehicle`, `scenario`, `ocr_burst_frames`, `min_votes`, `vote_window`, `fallback_ocr_enabled`
- `ocr_only_after_crossing`, `log_ocr_metrics`, `log_crossing_metrics`

---

## 4. Opções CLI — alpr-tool preview

Subcomando principal para vídeo + detecção. Argumentos que sobrescrevem ou complementam o .conf:

| Argumento | Descrição |
|-----------|-----------|
| `--conf <path>` | Arquivo .conf |
| `--source <video\|device>` | Vídeo ou dispositivo |
| `--country <br\|...>` | País/região |
| `--log-file <path>` | Log |
| `--crossing-mode off\|motion` | Modo de crossing (tripwire) |
| `--line x1,y1,x2,y2` | Linha de crossing (pixels) |
| `--crossing-line-pct <0-100>` | Linha por percentual da altura |
| `--crossing-fallback-sec <sec>` | Fallback: liberar OCR após N s |
| `--ocr-only-after-crossing 0\|1` | Gate: OCR só após crossing (motion) |
| `--plates-only-past-line 0\|1` | Contar só placas além da linha |
| `--crossing-roi x,y,w,h` | ROI do crossing |
| `--alpr-roi x,y,w,h` | ROI do ALPR |
| `--motion-thresh N` | Limiar de movimento |
| `--motion-min-area N` | Área mínima |
| `--motion-min-ratio R` | Razão mínima |
| `--motion-direction-filter 0\|1` | Filtrar direção |
| `--crossing-debounce N` | Debounce do crossing |
| `--crossing-arm-min-frames N` | Frames mínimos para arm |
| `--crossing-arm-min-ratio R` | Razão mínima para arm |
| `--vehicle car\|moto` | Veículo |
| `--scenario default\|garagem` | Cenário |
| `--profile default\|moto\|garagem` | Perfil (deprecated: usar vehicle+scenario) |
| `--max-seconds N` | Limite de tempo (s) |
| `--max-tracks N` | Máx. tracks |
| `--track-ttl-ms N` | TTL do track (ms) |
| `--log-plates 0\|1` | Log de placas |
| `--log-ocr-metrics 0\|1` | Métricas OCR |
| `--log-crossing-metrics 0\|1` | Métricas crossing |
| `--report-json <path>` | Relatório JSON |
| `--doctor` | Doctor/validação |

---

## 5. Presets em artifacts/configs

- `openalpr.default.conf` — carro, default
- `openalpr.moto.conf` — moto
- `openalpr.garagem.conf` — garagem
- `openalpr.moto_garagem.conf` — moto + garagem
- `openalpr.br.conf` / `openalpr.br2.conf` — país
- `openalpr.performance.conf` — performance
- `openalpr.full.conf` — referência completa

---

## 6. Resumo para um configurador web

Para uma interface web (ex.: Crow + Tailwind) é útil expor:

1. **Config principal**: runtime_dir, country, detector_type, detector, skip_detection, vehicle, scenario, ocr_burst_frames, vote_window, min_votes, fallback_ocr_enabled, enable_roi, roi_*, preproc_*, br_hybrid_*, debug_*.
2. **Preview (tool)**: source, crossing_mode, line / crossing_line_pct, ocr_only_after_crossing, plates_only_past_line, crossing_fallback_sec, vehicle, scenario, max_seconds.
3. **Presets**: listar e aplicar um dos arquivos em `artifacts/configs/`.
4. **Teste**: comando sugerido `alpr-tool preview --conf <path> --country br --source <video> [--plates-only-past-line 1] ...`.

Este estudo cobre as configurações existentes no código; novas chaves podem ser adicionadas no core ou no tool conforme a evolução do projeto.
