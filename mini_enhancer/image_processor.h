#ifndef MINI_ENHANCER_IMAGE_PROCESSOR_H
#define MINI_ENHANCER_IMAGE_PROCESSOR_H

#include <string>

enum class EnhanceTarget {
  General,     // nitidez visível
  AlprOcr,     // só bilateral leve — mais seguro para OCR em imagens já razoáveis
  AlprOcrMax   // CLAHE + bilateral + unsharp leve — só para frames muito escuros/ruidosos
};

// CPU only, no GPU. Returns empty string on failure.
std::string enhanceImage(const std::string& inputPath,
                         EnhanceTarget target = EnhanceTarget::General);

#endif
