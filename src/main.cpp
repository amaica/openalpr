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

#include <cstdio>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <signal.h>
#include <poll.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "tclap/CmdLine.h"
#include "support/filesystem.h"
#include "support/timing.h"
#include "support/platform.h"
#include "video/videobuffer.h"
#include "motiondetector.h"
#include "alpr.h"
#include "recognition_worker_process.h"

using namespace alpr;

const std::string MAIN_WINDOW_NAME = "ALPR main window";

const bool SAVE_LAST_VIDEO_STILL = false;
const std::string LAST_VIDEO_STILL_LOCATION = "/tmp/laststill.jpg";
const std::string WEBCAM_PREFIX = "/dev/video";
MotionDetector motiondetector;
bool do_motiondetection = true;

/** Function Headers */
bool detectandshow(Alpr* alpr, cv::Mat frame, std::string region, bool writeJson);
void print_results(const AlprResults& results, bool writeJson);
int processImagesParallel(const std::vector<std::string>& filenames, const std::string& country, const std::string& configFile, bool detectRegion, const std::string& templatePattern, int topn, bool debug_mode, bool outputJson, int jobs);
bool is_supported_image(std::string image_file);

bool measureProcessingTime = false;
std::string templatePattern;

// This boolean is set to false when the user hits terminates (e.g., CTRL+C )
// so we can end infinite loops for things like video processing.
bool program_active = true;

int main( int argc, const char** argv )
{
  std::vector<std::string> filenames;
  std::string configFile = "";
  bool outputJson = false;
  int seektoms = 0;
  bool detectRegion = false;
  std::string country;
  int topn;
  bool debug_mode = false;
  int jobs = 1;

  TCLAP::CmdLine cmd("OpenAlpr Command Line Utility", ' ', Alpr::getVersion());

  TCLAP::UnlabeledMultiArg<std::string>  fileArg( "image_file", "Image containing license plates", true, "", "image_file_path"  );

  
  TCLAP::ValueArg<std::string> countryCodeArg("c","country","Country code to identify (either us for USA or eu for Europe).  Default=us",false, "us" ,"country_code");
  TCLAP::ValueArg<int> seekToMsArg("","seek","Seek to the specified millisecond in a video file. Default=0",false, 0 ,"integer_ms");
  TCLAP::ValueArg<std::string> configFileArg("","config","Path to the openalpr.conf file",false, "" ,"config_file");
  TCLAP::ValueArg<std::string> templatePatternArg("p","pattern","Attempt to match the plate number against a plate pattern (e.g., md for Maryland, ca for California)",false, "" ,"pattern code");
  TCLAP::ValueArg<int> topNArg("n","topn","Max number of possible plate numbers to return.  Default=10",false, 10 ,"topN");
  TCLAP::ValueArg<int> jobsArg("","jobs","Number of parallel worker processes for image files.  Default=1 (synchronous)",false, 1 ,"jobs");

  TCLAP::SwitchArg jsonSwitch("j","json","Output recognition results in JSON format.  Default=off", cmd, false);
  TCLAP::SwitchArg debugSwitch("","debug","Enable debug output.  Default=off", cmd, false);
  TCLAP::SwitchArg detectRegionSwitch("d","detect_region","Attempt to detect the region of the plate image.  [Experimental]  Default=off", cmd, false);
  TCLAP::SwitchArg clockSwitch("","clock","Measure/print the total time to process image and all plates.  Default=off", cmd, false);
  TCLAP::SwitchArg motiondetect("", "motion", "Use motion detection on video file or stream.  Default=off", cmd, false);

  try
  {
    cmd.add( templatePatternArg );
    cmd.add( seekToMsArg );
    cmd.add( topNArg );
    cmd.add( jobsArg );
    cmd.add( configFileArg );
    cmd.add( fileArg );
    cmd.add( countryCodeArg );

    
    if (cmd.parse( argc, argv ) == false)
    {
      // Error occurred while parsing.  Exit now.
      return 1;
    }

    filenames = fileArg.getValue();

    country = countryCodeArg.getValue();
    seektoms = seekToMsArg.getValue();
    outputJson = jsonSwitch.getValue();
    debug_mode = debugSwitch.getValue();
    configFile = configFileArg.getValue();
    detectRegion = detectRegionSwitch.getValue();
    templatePattern = templatePatternArg.getValue();
    topn = topNArg.getValue();
    measureProcessingTime = clockSwitch.getValue();
	do_motiondetection = motiondetect.getValue();
    jobs = jobsArg.getValue();
  }
  catch (TCLAP::ArgException &e)    // catch any exceptions
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return 1;
  }

  // Fast path: parallel processing for lists of image files only.
  bool parallelEligible = jobs > 1;
  if (parallelEligible)
  {
    for (unsigned int i = 0; i < filenames.size(); i++)
    {
      const std::string& filename = filenames[i];
      if (!is_supported_image(filename) || filename == "-" || filename == "stdin" ||
          startsWith(filename, "http://") || startsWith(filename, "https://") ||
          filename == "webcam" || DirectoryExists(filename.c_str()))
      {
        parallelEligible = false;
        break;
      }
    }
  }

  if (parallelEligible)
  {
    return processImagesParallel(filenames, country, configFile, detectRegion, templatePattern, topn, debug_mode, outputJson, jobs);
  }
  else if (jobs > 1)
  {
    std::cerr << "Parallel mode (--jobs) is only supported for image file inputs. Running sequentially." << std::endl;
  }
  
  cv::Mat frame;

  Alpr alpr(country, configFile);
  alpr.setTopN(topn);
  
  if (debug_mode)
  {
    alpr.getConfig()->setDebug(true);
  }

  if (detectRegion)
    alpr.setDetectRegion(detectRegion);

  if (templatePattern.empty() == false)
    alpr.setDefaultRegion(templatePattern);

  if (alpr.isLoaded() == false)
  {
    std::cerr << "Error loading OpenALPR" << std::endl;
    return 1;
  }

  for (unsigned int i = 0; i < filenames.size(); i++)
  {
    std::string filename = filenames[i];

    if (filename == "-")
    {
      std::vector<uchar> data;
      int c;

      while ((c = fgetc(stdin)) != EOF)
      {
        data.push_back((uchar) c);
      }

      frame = cv::imdecode(cv::Mat(data), 1);
      if (!frame.empty())
      {
        detectandshow(&alpr, frame, "", outputJson);
      }
      else
      {
        std::cerr << "Image invalid: " << filename << std::endl;
      }
    }
    else if (filename == "stdin")
    {
      std::string filename;
      while (std::getline(std::cin, filename))
      {
        if (fileExists(filename.c_str()))
        {
          frame = cv::imread(filename);
          detectandshow(&alpr, frame, "", outputJson);
        }
        else
        {
          std::cerr << "Image file not found: " << filename << std::endl;
        }

      }
    }
    else if (filename == "webcam" || startsWith(filename, WEBCAM_PREFIX))
    {
      int webcamnumber = 0;
      
      // If they supplied "/dev/video[number]" parse the "number" here
      if(startsWith(filename, WEBCAM_PREFIX) && filename.length() > WEBCAM_PREFIX.length())
      {
        webcamnumber = atoi(filename.substr(WEBCAM_PREFIX.length()).c_str());
      }
      
      int framenum = 0;
      cv::VideoCapture cap(webcamnumber);
      if (!cap.isOpened())
      {
        std::cerr << "Error opening webcam" << std::endl;
        return 1;
      }

      while (cap.read(frame))
      {
        if (framenum == 0)
          motiondetector.ResetMotionDetection(&frame);
        detectandshow(&alpr, frame, "", outputJson);
        sleep_ms(10);
        framenum++;
      }
    }
    else if (startsWith(filename, "http://") || startsWith(filename, "https://"))
    {
      int framenum = 0;

      VideoBuffer videoBuffer;

      videoBuffer.connect(filename, 5);

      cv::Mat latestFrame;

      while (program_active)
      {
        std::vector<cv::Rect> regionsOfInterest;
        int response = videoBuffer.getLatestFrame(&latestFrame, regionsOfInterest);

        if (response != -1)
        {
          if (framenum == 0)
            motiondetector.ResetMotionDetection(&latestFrame);
          detectandshow(&alpr, latestFrame, "", outputJson);
        }

        // Sleep 10ms
        sleep_ms(10);
        framenum++;
      }

      videoBuffer.disconnect();

      std::cout << "Video processing ended" << std::endl;
    }
    else if (hasEndingInsensitive(filename, ".avi") || hasEndingInsensitive(filename, ".mp4") ||
                                                       hasEndingInsensitive(filename, ".webm") ||
                                                       hasEndingInsensitive(filename, ".flv") || hasEndingInsensitive(filename, ".mjpg") ||
                                                       hasEndingInsensitive(filename, ".mjpeg") ||
             hasEndingInsensitive(filename, ".mkv")
        )
    {
      if (fileExists(filename.c_str()))
      {
        int framenum = 0;

        cv::VideoCapture cap = cv::VideoCapture();
        cap.open(filename);
        cap.set(cv::CAP_PROP_POS_MSEC, seektoms);

        while (cap.read(frame))
        {
          if (SAVE_LAST_VIDEO_STILL)
          {
            cv::imwrite(LAST_VIDEO_STILL_LOCATION, frame);
          }
          if (!outputJson)
            std::cout << "Frame: " << framenum << std::endl;
          
          if (framenum == 0)
            motiondetector.ResetMotionDetection(&frame);
          detectandshow(&alpr, frame, "", outputJson);
          //create a 1ms delay
          sleep_ms(1);
          framenum++;
        }
      }
      else
      {
        std::cerr << "Video file not found: " << filename << std::endl;
      }
    }
    else if (is_supported_image(filename))
    {
      if (fileExists(filename.c_str()))
      {
        frame = cv::imread(filename);

        bool plate_found = detectandshow(&alpr, frame, "", outputJson);

        if (!plate_found && !outputJson)
          std::cout << "No license plates found." << std::endl;
      }
      else
      {
        std::cerr << "Image file not found: " << filename << std::endl;
      }
    }
    else if (DirectoryExists(filename.c_str()))
    {
      std::vector<std::string> files = getFilesInDir(filename.c_str());

      std::sort(files.begin(), files.end(), stringCompare);

      for (int i = 0; i < files.size(); i++)
      {
        if (is_supported_image(files[i]))
        {
          std::string fullpath = filename + "/" + files[i];
          std::cout << fullpath << std::endl;
          frame = cv::imread(fullpath.c_str());
          if (detectandshow(&alpr, frame, "", outputJson))
          {
            //while ((char) cv::waitKey(50) != 'c') { }
          }
          else
          {
            //cv::waitKey(50);
          }
        }
      }
    }
    else
    {
      bool img = is_supported_image(filename);
      bool dir = DirectoryExists(filename.c_str());
      bool vid = hasEndingInsensitive(filename, ".mp4") || hasEndingInsensitive(filename, ".avi") ||
                 hasEndingInsensitive(filename, ".mkv") || hasEndingInsensitive(filename, ".webm") ||
                 hasEndingInsensitive(filename, ".flv") || hasEndingInsensitive(filename, ".mjpg") ||
                 hasEndingInsensitive(filename, ".mjpeg");
      std::cerr << "Unknown file type: " << filename << " img=" << img << " dir=" << dir << " vid=" << vid << std::endl;
      return 1;
    }
  }

  return 0;
}

bool is_supported_image(std::string image_file)
{
  return (hasEndingInsensitive(image_file, ".png") || hasEndingInsensitive(image_file, ".jpg") || 
	  hasEndingInsensitive(image_file, ".tif") || hasEndingInsensitive(image_file, ".bmp") ||  
	  hasEndingInsensitive(image_file, ".jpeg") || hasEndingInsensitive(image_file, ".gif"));
}


void print_results(const AlprResults& results, bool writeJson)
{
  if (writeJson)
  {
    std::cout << Alpr::toJson(results) << std::endl;
    return;
  }

  for (int i = 0; i < results.plates.size(); i++)
  {
    std::cout << "plate" << i << ": " << results.plates[i].topNPlates.size() << " results";
    if (measureProcessingTime)
      std::cout << " -- Processing Time = " << results.plates[i].processing_time_ms << "ms.";
    std::cout << std::endl;

    if (results.plates[i].regionConfidence > 0)
      std::cout << "State ID: " << results.plates[i].region << " (" << results.plates[i].regionConfidence << "% confidence)" << std::endl;
    
    for (int k = 0; k < results.plates[i].topNPlates.size(); k++)
    {
      std::string no_newline = results.plates[i].topNPlates[k].characters;
      std::replace(no_newline.begin(), no_newline.end(), '\n','-');
      
      std::cout << "    - " << no_newline << "\t confidence: " << results.plates[i].topNPlates[k].overall_confidence;
      if (templatePattern.size() > 0 || results.plates[i].regionConfidence > 0)
        std::cout << "\t pattern_match: " << results.plates[i].topNPlates[k].matches_template;
      
      std::cout << std::endl;
    }
  }
}


bool detectandshow( Alpr* alpr, cv::Mat frame, std::string region, bool writeJson)
{

  timespec startTime;
  getTimeMonotonic(&startTime);

  std::vector<AlprRegionOfInterest> regionsOfInterest;
  if (do_motiondetection)
  {
	  cv::Rect rectan = motiondetector.MotionDetect(&frame);
	  if (rectan.width>0) regionsOfInterest.push_back(AlprRegionOfInterest(rectan.x, rectan.y, rectan.width, rectan.height));
  }
  else regionsOfInterest.push_back(AlprRegionOfInterest(0, 0, frame.cols, frame.rows));
  AlprResults results;
  if (regionsOfInterest.size()>0) results = alpr->recognize(frame.data, frame.elemSize(), frame.cols, frame.rows, regionsOfInterest);

  timespec endTime;
  getTimeMonotonic(&endTime);
  double totalProcessingTime = diffclock(startTime, endTime);
  if (measureProcessingTime)
    std::cout << "Total Time to process image: " << totalProcessingTime << "ms." << std::endl;
  
  
  if (writeJson)
  {
    print_results(results, writeJson);
  }
  else
  {
    print_results(results, writeJson);
  }



  return results.plates.size() > 0;
}


int processImagesParallel(const std::vector<std::string>& filenames, const std::string& country, const std::string& configFile, bool detectRegion, const std::string& templatePatternParam, int topn, bool debug_mode, bool outputJson, int jobs)
{
  if (filenames.size() == 0)
    return 0;

  int workerCount = std::max(1, std::min(static_cast<int>(filenames.size()), std::max(1, jobs)));
  RecognitionWorkerProcess::Params params;
  params.country = country;
  params.configFile = configFile;
  params.templatePattern = templatePatternParam;
  params.topn = topn;
  params.detectRegion = detectRegion;
  params.debug = debug_mode;
  params.measureProcessingTime = measureProcessingTime;

  struct WorkerState
  {
    RecognitionWorkerProcess proc;
    bool busy;
    std::string currentFile;
    WorkerState(const RecognitionWorkerProcess::Params& p) : proc(p), busy(false) {}
  };

  std::vector<WorkerState> workers;
  for (int i = 0; i < workerCount; i++)
  {
    workers.push_back(WorkerState(params));
    if (!workers.back().proc.start())
    {
      std::cerr << "Failed to start worker process " << i << std::endl;
      return 1;
    }
  }

  size_t nextFileIdx = 0;
  int active = 0;

  while (nextFileIdx < filenames.size() || active > 0)
  {
    for (int i = 0; i < workerCount && nextFileIdx < filenames.size(); i++)
    {
      if (workers[i].busy)
        continue;

      std::string file = filenames[nextFileIdx];
      nextFileIdx++; // always advance to avoid infinite retry on a bad file
      if (!workers[i].proc.sendJob(file))
      {
        std::cerr << "Failed to send job to worker" << std::endl;
        continue;
      }
      workers[i].busy = true;
      workers[i].currentFile = file;
      active++;
    }

    if (active == 0)
      break;

    std::vector<pollfd> fds;
    std::vector<int> idxmap;
    for (int i = 0; i < workerCount; i++)
    {
      if (!workers[i].busy) continue;
      pollfd pfd;
      pfd.fd = workers[i].proc.readFd();
      pfd.events = POLLIN;
      pfd.revents = 0;
      fds.push_back(pfd);
      idxmap.push_back(i);
    }

    int ret = poll(fds.data(), fds.size(), 500);
    if (ret <= 0)
      continue;

    for (size_t f = 0; f < fds.size(); f++)
    {
      if (!(fds[f].revents & POLLIN))
        continue;
      int widx = idxmap[f];
      std::string imagePath;
      std::string json;
      if (!workers[widx].proc.readResult(imagePath, json))
      {
        workers[widx].busy = false;
        active--;
        continue;
      }
      workers[widx].busy = false;
      active--;

      AlprResults results = Alpr::fromJson(json);

      if (outputJson)
        print_results(results, true);
      else
      {
        if (results.plates.size() == 0)
          std::cout << "No license plates found for " << imagePath << "." << std::endl;
        else
          print_results(results, false);
      }
    }
  }

  for (int i = 0; i < workerCount; i++)
    workers[i].proc.stop();

  return 0;
}

