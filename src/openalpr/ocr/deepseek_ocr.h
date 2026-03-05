/*
 * Copyright (c) 2015 OpenALPR Technology, Inc.
 * Open source Automated License Plate Recognition [http://www.openalpr.com]
 *
 * This file is part of OpenALPR.
 *
 * OpenALPR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License
 * version 3 as published by the Free Software Foundation
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPENALPR_DEEPSEEK_OCR_H
#define OPENALPR_DEEPSEEK_OCR_H

#include "ocr.h"

namespace alpr
{

  class DeepSeekOCR : public OCR
  {
  public:
    DeepSeekOCR(Config* config);
    virtual ~DeepSeekOCR();

  protected:
    virtual std::vector<OcrChar> recognize_line(int line_index, PipelineData* pipeline_data);
    virtual void segment(PipelineData* pipeline_data);

  private:
    std::string performDeepSeekRequest(const std::vector<unsigned char>& imageBuf);
    std::vector<OcrChar> parseDeepSeekResponse(const std::string& response);
  };

}

#endif // OPENALPR_DEEPSEEK_OCR_H
