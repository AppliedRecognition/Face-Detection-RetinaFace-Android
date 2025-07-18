//
// Created by Jakub Dolejs on 11/07/2025.
//

#include "Postprocessing.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>

namespace verid {

    Postprocessing::Postprocessing(int imageWidth, int imageHeight)
                : imageWidth(imageWidth), imageHeight(imageHeight), scoreThreshold(0.3f)
    {
        std::vector<std::vector<int>> minSizes = { {16, 32}, {64, 128}, {256, 512} };
        std::vector<int> steps = { 8, 16, 32 };

        int predictionCount = 0;
        for (size_t k = 0; k < steps.size(); ++k) {
            int step = steps[k];
            int fH = static_cast<int>(std::ceil(static_cast<float>(imageHeight) / step));
            int fW = static_cast<int>(std::ceil(static_cast<float>(imageWidth) / step));
            predictionCount += fH * fW * static_cast<int>(minSizes[k].size());
        }
        int boxCount = 4;
        int scoreCount = 2;
        int landmarkCount = 10;

        boxIndices.resize(boxCount);
        for (int i = 0; i < boxCount; ++i) {
            for (int j = i; j < predictionCount * boxCount; j += boxCount)
                boxIndices[i].push_back(j);
        }

        scoreIndices.resize(scoreCount);
        for (int i = 0; i < scoreCount; ++i) {
            for (int j = i; j < predictionCount * scoreCount; j += scoreCount)
                scoreIndices[i].push_back(j);
        }

        landmarkIndices.resize(landmarkCount);
        for (int i = 0; i < landmarkCount; ++i) {
            for (int j = i; j < predictionCount * landmarkCount; j += landmarkCount)
                landmarkIndices[i].push_back(j);
        }

        priors = generatePriors(minSizes, steps);
    }

    std::vector<DetectionBox> Postprocessing::decode(
            const std::vector<float>& boxesArray,
            const std::vector<float>& scoresArray,
            const std::vector<float>& landmarkArray)
    {
        int count = scoresArray.size() / 2;
        std::vector<float> confScores(count);
        for (int i = 0; i < count; ++i) {
            confScores[i] = scoresArray[scoreIndices[1][i]];
        }

        std::vector<int> retainedIndices;
        for (int i = 0; i < count; ++i) {
            if (confScores[i] >= scoreThreshold)
                retainedIndices.push_back(i);
        }
        if (retainedIndices.empty()) return {};

        const auto& cx = priors[0], &cy = priors[1], &pw = priors[2], &ph = priors[3];
        std::vector<DetectionBox> detections;

        for (int idx : retainedIndices) {
            float dx = boxesArray[boxIndices[0][idx]];
            float dy = boxesArray[boxIndices[1][idx]];
            float dw = boxesArray[boxIndices[2][idx]];
            float dh = boxesArray[boxIndices[3][idx]];

            float adjX = cx[idx] + 0.1f * dx * pw[idx];
            float adjY = cy[idx] + 0.1f * dy * ph[idx];
            float expW = pw[idx] * std::exp(0.2f * dw);
            float expH = ph[idx] * std::exp(0.2f * dh);

            float x1 = adjX - expW / 2.0f;
            float y1 = adjY - expH / 2.0f;

            Rect rect = { x1 * imageWidth, y1 * imageHeight, expW * imageWidth, expH * imageHeight };

            std::vector<Point> landmarkPoints;
            for (int i = 0; i < 5; ++i) {
                float lx = landmarkArray[landmarkIndices[2 * i][idx]];
                float ly = landmarkArray[landmarkIndices[2 * i + 1][idx]];

                float pointX = cx[idx] + 0.1f * lx * pw[idx];
                float pointY = cy[idx] + 0.1f * ly * ph[idx];
                landmarkPoints.push_back({ pointX * imageWidth, pointY * imageHeight });
            }

            EulerAngle angle = calculateFaceAngle(
                    landmarkPoints[0], landmarkPoints[1],
                    landmarkPoints[2], landmarkPoints[3], landmarkPoints[4]
            );

            detections.push_back({ confScores[idx], rect, landmarkPoints, angle, confScores[idx] });
        }

        return detections;
    }

    std::vector<DetectionBox> Postprocessing::nonMaxSuppression(
            std::vector<DetectionBox>& boxes, float iouThreshold, int limit)
    {
        std::vector<DetectionBox> selected;
        std::sort(boxes.begin(), boxes.end(),
                  [](const DetectionBox& a, const DetectionBox& b) { return a.score > b.score; });

        for (const auto& box : boxes) {
            if ((int)selected.size() >= limit) break;
            bool keep = true;
            for (const auto& sel : selected) {
                if (iou(sel.bounds, box.bounds) >= iouThreshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) selected.push_back(box);
        }
        return selected;
    }



    [[nodiscard]] std::vector<std::vector<float>> Postprocessing::generatePriors(
            const std::vector<std::vector<int>>& minSizes,
            const std::vector<int>& steps) const {
        std::vector<std::vector<float>> anchors(4);

        for (size_t k = 0; k < steps.size(); ++k) {
            int step = steps[k];
            int fH = (int)std::ceil((float)imageHeight / step);
            int fW = (int)std::ceil((float)imageWidth / step);

            for (int i = 0; i < fH; ++i) {
                for (int j = 0; j < fW; ++j) {
                    for (int minSize : minSizes[k]) {
                        float s_kx = (float)minSize / imageWidth;
                        float s_ky = (float)minSize / imageHeight;
                        float cx = (j + 0.5f) * step / imageWidth;
                        float cy = (i + 0.5f) * step / imageHeight;

                        anchors[0].push_back(cx);
                        anchors[1].push_back(cy);
                        anchors[2].push_back(s_kx);
                        anchors[3].push_back(s_ky);
                    }
                }
            }
        }
        return anchors;
    }

    EulerAngle Postprocessing::calculateFaceAngle(const Point& leftEye, const Point& rightEye,
                                         const Point& noseTip, const Point& leftMouth,
                                         const Point& rightMouth)
    {
        float dx = rightEye.x - leftEye.x;
        float dy = rightEye.y - leftEye.y;
        float roll = std::atan2(dy, dx) * 180.0f / M_PI;

        Point eyeCenter = { (leftEye.x + rightEye.x) / 2.0f, (leftEye.y + rightEye.y) / 2.0f };
        Point mouthCenter = { (leftMouth.x + rightMouth.x) / 2.0f, (leftMouth.y + rightMouth.y) / 2.0f };

        float interocular = rightEye.x - leftEye.x;
        float noseOffset = noseTip.x - eyeCenter.x;
        float yaw = std::atan2(noseOffset, interocular) * 180.0f / M_PI * 1.2f;

        float verticalFaceLength = mouthCenter.y - eyeCenter.y;
        float verticalNoseOffset = noseTip.y - eyeCenter.y;
        float pitchRatio = verticalNoseOffset / verticalFaceLength;
        float pitch = (0.5f - pitchRatio) * 90.0f;

        return { yaw, pitch, roll };
    }

    float Postprocessing::iou(const Rect& a, const Rect& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);

        float interW = std::max(0.0f, x2 - x1);
        float interH = std::max(0.0f, y2 - y1);
        float interArea = interW * interH;

        if (interArea <= 0.0f) return 0.0f;

        float unionArea = a.width * a.height + b.width * b.height - interArea;
        return interArea / unionArea;
    }

} // verid
