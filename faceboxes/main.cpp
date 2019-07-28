#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "faceboxDetector.h"

using namespace std;

int main(int argc, char** argv)
{
    // create class
    string imgPath;
    if  (argc = 1)
    {
        imgPath = "../../model/sample.jpg";
    }
    else if (argc = 2)
    {
        imgPath = argv[1];
    }
    string param = "../../model/facebox.param";
    string bin = "../../model/facebox.bin";

    Detector detector(param, bin);

    for	(int i = 0; i < 100; i++){

        cv::Mat m = cv::imread(imgPath.c_str());
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
            return -1;
        }

        int64 start = cv::getTickCount();
        std::vector<bbox> boxes;
        detector.Detect(m, boxes);
        double duration = (cv::getTickCount()-start)/ cv::getTickFrequency()*1000;
        printf("%f\n", duration);

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);
            cv::rectangle(m, rect, cv::Scalar(0, 0, 255), 3, 8, 0);
        }
        cv::imwrite("../../model/test.jpg", m);
    }
    return 0;
}

