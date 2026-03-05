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

#include "deepseek_ocr.h"
#include "config.h"
#include "segmentation/charactersegmenter.h"
#include <curl/curl.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cjson.h"
#include <cstdlib>

using namespace std;
using namespace cv;

namespace alpr
{
  static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
  {
      ((std::string*)userp)->append((char*)contents, size * nmemb);
      return size * nmemb;
  }

  // Helper for base64 encoding
  static const std::string base64_chars = 
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "abcdefghijklmnopqrstuvwxyz"
               "0123456789+/";

  static std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
      char_array_3[i++] = *(bytes_to_encode++);
      if (i == 3) {
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for(i = 0; (i <4) ; i++)
          ret += base64_chars[char_array_4[i]];
        i = 0;
      }
    }

    if (i)
    {
      for(j = i; j < 3; j++)
        char_array_3[j] = '\0';

      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (j = 0; (j < i + 1); j++)
        ret += base64_chars[char_array_4[j]];

      while((i++ < 3))
        ret += '=';
    }

    return ret;
  }

  DeepSeekOCR::DeepSeekOCR(Config* config) : OCR(config)
  {
  }

  DeepSeekOCR::~DeepSeekOCR()
  {
  }

  void DeepSeekOCR::segment(PipelineData* pipeline_data)
  {
    // Reuse the existing character segmenter to find lines
    CharacterSegmenter segmenter(pipeline_data);
    segmenter.segment();
  }

  std::vector<OcrChar> DeepSeekOCR::recognize_line(int line_idx, PipelineData* pipeline_data)
  {
    std::vector<OcrChar> chars;

    if (line_idx >= pipeline_data->textLines.size())
      return chars;

    // Get the line crop (TextLine has textArea, not rect — compute bounding rect)
    const std::vector<cv::Point>& area = pipeline_data->textLines[line_idx].textArea;
    if (area.empty()) return chars;
    cv::Rect lineRect = cv::boundingRect(area);
    
    // Safety check on coords
    lineRect = expandRect(lineRect, 0, 0, pipeline_data->grayImg.cols, pipeline_data->grayImg.rows);
    
    cv::Mat lineImg = pipeline_data->grayImg(lineRect);
    
    // Encode to JPG then Base64
    std::vector<uchar> buf;
    cv::imencode(".jpg", lineImg, buf);
    std::string base64_img = base64_encode(buf.data(), buf.size());

    // Send request
    std::string response = performDeepSeekRequest(buf); // Passing raw buf to helper if we want to change encoding later

    // Parse Response
    chars = parseDeepSeekResponse(response);

    return chars;
  }

  std::string DeepSeekOCR::performDeepSeekRequest(const std::vector<unsigned char>& imageBuf)
  {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if(curl) {
      const char* env_key = getenv("DEEPSEEK_API_KEY");
      std::string api_key = env_key ? std::string(env_key) : "";
      
      // Fallback: Check if user put it in config (assuming we added a generic property accessor or just use empty)
      if (api_key.empty()) {
          // Warning
          std::cerr << "DeepSeek API Key not found in environment (DEEPSEEK_API_KEY)." << std::endl;
          return "";
      }

      std::string url = "https://api.deepseek.com/v1/chat/completions"; 
      
      struct curl_slist *headers = NULL;
      headers = curl_slist_append(headers, "Content-Type: application/json");
      std::string auth_header = "Authorization: Bearer " + api_key;
      headers = curl_slist_append(headers, auth_header.c_str());

      // Construct JSON payload
      // Using cJSON for robustness
      cJSON *root = cJSON_CreateObject();
      cJSON_AddStringToObject(root, "model", "deepseek-vl"); // Assuming this is the model name for vision
      
      cJSON *messages = cJSON_CreateArray();
      cJSON_AddItemToObject(root, "messages", messages);
      
      cJSON *message = cJSON_CreateObject();
      cJSON_AddItemToArray(messages, message);
      cJSON_AddStringToObject(message, "role", "user");
      
      // Message content is array of text and image
      cJSON *content = cJSON_CreateArray();
      cJSON_AddItemToObject(message, "content", content);
      
      // Text prompt
      cJSON *textPart = cJSON_CreateObject();
      cJSON_AddStringToObject(textPart, "type", "text");
      cJSON_AddStringToObject(textPart, "text", "Read the license plate in this image. Return ONLY the uppercased alphanumeric text. No spaces, no hyphens.");
      cJSON_AddItemToArray(content, textPart);

      // Image part
      cJSON *imagePart = cJSON_CreateObject();
      cJSON_AddStringToObject(imagePart, "type", "image_url");
      cJSON *imageUrl = cJSON_CreateObject();
      std::string b64 = base64_encode(imageBuf.data(), imageBuf.size());
      std::string data_uri = "data:image/jpeg;base64," + b64;
      cJSON_AddStringToObject(imageUrl, "url", data_uri.c_str());
      cJSON_AddItemToObject(imagePart, "image_url", imageUrl);
      cJSON_AddItemToArray(content, imagePart);
      
      cJSON *max_tokens = cJSON_CreateNumber(50);
      cJSON_AddItemToObject(root, "max_tokens", max_tokens);

      char *jsonString = cJSON_PrintUnformatted(root);
      
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonString);
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
      curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10 seconds timeout

      res = curl_easy_perform(curl);
      if(res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

      cJSON_Delete(root);
      free(jsonString);
      curl_slist_free_all(headers);
      curl_easy_cleanup(curl);
    }
    return readBuffer;
  }

  std::vector<OcrChar> DeepSeekOCR::parseDeepSeekResponse(const std::string& response)
  {
    std::vector<OcrChar> chars;
    if (response.empty()) return chars;

    cJSON *json = cJSON_Parse(response.c_str());
    if (!json) {
        std::cerr << "Failed to parse DeepSeek JSON response" << std::endl;
        return chars;
    }

    // Extract content from choices[0].message.content
    cJSON *choices = cJSON_GetObjectItem(json, "choices");
    if (choices && cJSON_GetArraySize(choices) > 0) {
        cJSON *choice = cJSON_GetArrayItem(choices, 0);
        cJSON *message = cJSON_GetObjectItem(choice, "message");
        if (message) {
            cJSON *content = cJSON_GetObjectItem(message, "content");
            if (content && content->valuestring) {
                std::string text = content->valuestring;
                // Clean the text (remove newlines, spaces, non-alphanumeric)
                int index = 0;
                for (char c : text) {
                    if (isalnum(c)) {
                        OcrChar oc;
                        oc.letter = std::string(1, toupper(c));
                        oc.confidence = 90.0f; // Mock confidence since API might not give per-char
                        oc.char_index = index++;
                        chars.push_back(oc);
                    }
                }
            }
        }
    }
    
    cJSON_Delete(json);
    return chars;
  }

}
