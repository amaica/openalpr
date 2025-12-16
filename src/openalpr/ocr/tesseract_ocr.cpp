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

#include "tesseract_ocr.h"
#include "config.h"

#include "segmentation/charactersegmenter.h"
#include <numeric>

using namespace std;
using namespace cv;
using namespace tesseract;

namespace alpr
{

  TesseractOcr::TesseractOcr(Config* config)
  : OCR(config)
  {
    const string MINIMUM_TESSERACT_VERSION = "3.03";

    this->postProcessor.setConfidenceThreshold(config->postProcessMinConfidence, config->postProcessConfidenceSkipLevel);
    
    if (cmpVersion(tesseract.Version(), MINIMUM_TESSERACT_VERSION.c_str()) < 0)
    {
      std::cerr << "Warning: You are running an unsupported version of Tesseract." << endl;
      std::cerr << "Expecting at least " << MINIMUM_TESSERACT_VERSION << ", your version is: " << tesseract.Version() << endl;
    }

    string TessdataPrefix = config->getTessdataPrefix();
    if (cmpVersion(tesseract.Version(), "4.0.0") >= 0)
      TessdataPrefix += "tessdata/";    

    // Tesseract requires the prefix directory to be set as an env variable
    tesseract.Init(TessdataPrefix.c_str(), config->ocrLanguage.c_str() 	);
    tesseract.SetVariable("save_blob_choices", "T");
    tesseract.SetVariable("debug_file", "/dev/null");
    tesseract.SetPageSegMode(PSM_SINGLE_CHAR);
  }

  TesseractOcr::~TesseractOcr()
  {
    tesseract.End();
  }
  
  std::vector<OcrChar> TesseractOcr::recognize_line(int line_idx, PipelineData* pipeline_data) {

    const int SPACE_CHAR_CODE = 32;
    
    std::vector<OcrChar> best_chars;
    double best_score = -1.0;
    
    auto runPass = [&](const cv::Mat& src, double scale, int passIndex, int threshIndex)->std::pair<std::vector<OcrChar>, double> {
      cv::Mat working;
      src.copyTo(working);
      bitwise_not(working, working);
      tesseract.SetImage((uchar*) working.data,
                          working.size().width, working.size().height,
                          working.channels(), working.step1());
      std::vector<OcrChar> chars;
      int absolute_charpos = 0;
      for (unsigned int j = 0; j < pipeline_data->charRegions[line_idx].size(); j++)
      {
        cv::Rect base = pipeline_data->charRegions[line_idx][j];
        cv::Rect scaledRect(cvRound(base.x * scale), cvRound(base.y * scale),
                            cvRound(base.width * scale), cvRound(base.height * scale));
        Rect expandedRegion = expandRect(scaledRect, 2, 2, working.cols, working.rows);

        tesseract.SetRectangle(expandedRegion.x, expandedRegion.y, expandedRegion.width, expandedRegion.height);
        tesseract.Recognize(NULL);

        tesseract::ResultIterator* ri = tesseract.GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;
        do
        {
          if (ri->Empty(level)) continue;
          
          const char* symbol = ri->GetUTF8Text(level);
          float conf = ri->Confidence(level);

          bool dontcare;
          int fontindex = 0;
          int pointsize = 0;
          const char* fontName = ri->WordFontAttributes(&dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &pointsize, &fontindex);

          if(symbol != 0 && symbol[0] != SPACE_CHAR_CODE && pointsize >= config->ocrMinFontSize)
          {
            OcrChar c;
            c.char_index = absolute_charpos;
            c.confidence = conf;
            c.letter = string(symbol);
            chars.push_back(c);

            if (this->config->debugOcr)
              printf("charpos%d line%d: pass %d (thresh %d) symbol %s, conf: %f font: %s (index %d) size %dpx", absolute_charpos, line_idx, passIndex, threshIndex, symbol, conf, fontName, fontindex, pointsize);

            bool indent = false;
            tesseract::ChoiceIterator ci(*ri);
            do
            {
              const char* choice = ci.GetUTF8Text();
              
              OcrChar c2;
              c2.char_index = absolute_charpos;
              c2.confidence = ci.Confidence();
              c2.letter = string(choice);
              
              if (string(symbol) != string(choice))
                chars.push_back(c2);
              else
              {
                chars.push_back(c2);
              }
              if (this->config->debugOcr)
              {
                if (indent) printf("\t\t ");
                printf("\t- ");
                printf("%s conf: %f\n", choice, ci.Confidence());
              }

              indent = true;
            }
            while(ci.Next());

          }

          if (this->config->debugOcr)
            printf("---------------------------------------------\n");

          delete[] symbol;
        }
        while((ri->Next(level)));

        delete ri;

        absolute_charpos++;
      }
      double score = 0.0;
      for (const auto& c : chars) score += c.confidence;
      return std::make_pair(chars, score);
    };

    auto buildPasses = [&](const cv::Mat& base, std::vector<std::pair<cv::Mat,double>>& out){
      out.clear();
      out.push_back(std::make_pair(base, 1.0));
      bool isMoto = (config->vehicle == "moto");
      bool isGaragem = (config->scenario == "garagem");
      bool applyUpsample = config->motoUpsample || isMoto || isGaragem;
      double upScale = (config->motoUpsampleScale > 0.0) ? config->motoUpsampleScale : 2.0;
      if (applyUpsample && upScale != 1.0) {
        cv::Mat up;
        cv::resize(base, up, cv::Size(), upScale, upScale, cv::INTER_CUBIC);
        out.push_back(std::make_pair(up, upScale));
      }

      if (isMoto || isGaragem) {
        cv::Mat claheImg;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        clahe->apply(base, claheImg);
        cv::Mat adapt;
        cv::adaptiveThreshold(claheImg, adapt, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, 5);
        out.push_back(std::make_pair(adapt, 1.0));

        if (isGaragem) {
          cv::Mat blurred;
          cv::GaussianBlur(base, blurred, cv::Size(3,3), 0);
          cv::Mat sharpened;
          cv::addWeighted(base, 1.5, blurred, -0.5, 0, sharpened);
          cv::Mat adapt2;
          cv::adaptiveThreshold(sharpened, adapt2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 17, 7);
          out.push_back(std::make_pair(adapt2, 1.0));
        }
      }
    };

    for (unsigned int i = 0; i < pipeline_data->thresholds.size(); i++)
    {
      std::vector<std::pair<cv::Mat,double>> passes;
      buildPasses(pipeline_data->thresholds[i], passes);
      for (size_t p = 0; p < passes.size(); ++p) {
        pipeline_data->ocr_passes_total++;
        auto res = runPass(passes[p].first, passes[p].second, static_cast<int>(p), static_cast<int>(i));
        if (res.second > best_score) {
          best_score = res.second;
          best_chars = res.first;
        }
      }
    }
    
    return best_chars;
  }
  void TesseractOcr::segment(PipelineData* pipeline_data) {

    CharacterSegmenter segmenter(pipeline_data);
    segmenter.segment();
  }


}
