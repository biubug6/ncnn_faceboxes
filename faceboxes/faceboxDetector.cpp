#include <algorithm>
//#include "omp.h"
#include "faceboxDetector.h"

Detector::Detector():
        _nms(0.3),
        _threshold(0.5),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
}

inline void Detector::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Detector::Detector(const std::string &model_param, const std::string &model_bin):
        _nms(0.3),
        _threshold(0.5),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
    Init(model_param, model_bin);
}

void Detector::Init(const std::string &model_param, const std::string &model_bin)
{
    Net->load_param(model_param.c_str());
    Net->load_model(model_bin.c_str());
}

void Detector::Detect(cv::Mat& bgr, std::vector<bbox>& boxes)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, bgr.cols, bgr.rows);
    in.substract_mean_normalize(_mean_val, 0);

    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
    ex.input(0, in);
    // int final_idx = facebox.blobs.size() - 1;
    ncnn::Mat out;
    ex.extract("out", out);

    std::vector<box> anchor;
    create_anchor(anchor, bgr.cols, bgr.rows);

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr+5) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr + 5);
            total_box.push_back(result);
        }
        ptr = ptr + 6;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);
    printf("%d\n", (int)total_box.size());

    for (int j = 0; j < total_box.size(); ++j)
    {
        boxes.push_back(total_box[j]);
    }
}


inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detector::SetDefaultParams(){
    _nms = 0.3;
    _threshold = 0.5;
    _mean_val[0] = 104.f;
    _mean_val[1] = 117.f;
    _mean_val[2] = 123.f;
    Net = nullptr;

}

Detector::~Detector(){
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h)
{
    int num_boxes = 21 * ceil(w / 32.) * ceil(h / 32.) +
                    ceil(w / 64.) * ceil(h / 64.) +
                    ceil(w / 128.) * ceil(h / 128.);
    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {32, 64, 128};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));

        //
        std::vector<int> tmp;
        if (i == 0){
            for (int j = 0; j < 3; ++j) {
                tmp.push_back(pow(2,j) * 32);
            }
        } else
            tmp.push_back(i*256);
        min_sizes[i] = tmp;
    }

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    if (min_size[l] == 32){
                        for (int m = 0; m < 4; ++m) {
                            float dense_y = (0.25*m + i)*steps[k]/h;
                            for (int n = 0; n < 4; ++n) {
                                float dense_x = (0.25*n + j)*steps[k]/w;
                                box axil = {dense_x, dense_y, s_kx, s_ky};
                                anchor.push_back(axil);

                            }
                        }
                    } else if (min_size[l] == 64) {
                        for (int m = 0; m < 2; ++m) {
                            float dense_y = (0.5*m + i)*steps[k]/h;
                            for (int n = 0; n < 2; ++n) {
                                float dense_x = (0.5*n + j)*steps[k]/w;
                                box axil = {dense_x, dense_y, s_kx, s_ky};
                                anchor.push_back(axil);
                            }
                        }
                    } else {
                        float cx = (j + 0.5) * steps[k]/w;
                        float cy = (i + 0.5) * steps[k]/h;
                        box axil = {cx, cy, s_kx, s_ky};
                        anchor.push_back(axil);
                    }
                }
            }
        }

    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}