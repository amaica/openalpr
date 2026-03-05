#include "deepseek_ocr.h"
#include "tesseract_ocr.h"

namespace alpr
{
  OCR* createOcr(Config* config, std::string type)
  {
    if (type == "deepseek")
      return new DeepSeekOCR(config);

    return new TesseractOcr(config);
  }

}

