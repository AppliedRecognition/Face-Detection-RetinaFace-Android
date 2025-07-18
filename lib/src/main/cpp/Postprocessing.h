#ifndef FACE_DETECTION_POSTPROCESSING_H
#define FACE_DETECTION_POSTPROCESSING_H

#include <vector>

namespace verid {

    struct Point {
        float x, y;
    };

    struct Rect {
        float x, y, width, height;
    };

    struct EulerAngle {
        float yaw = 0, pitch = 0, roll = 0;
    };

    struct DetectionBox {
        float score;
        Rect bounds;
        std::vector<Point> landmarks;
        EulerAngle angle;
        float quality;
    };

    class Postprocessing {
    public:
        explicit Postprocessing(int imageWidth, int imageHeight);
        std::vector<DetectionBox> decode(
                const std::vector<float>& boxesArray,
                const std::vector<float>& scoresArray,
                const std::vector<float>& landmarkArray);
        static std::vector<DetectionBox> nonMaxSuppression(
                std::vector<DetectionBox>& boxes, float iouThreshold, int limit);
    private:
        int imageWidth, imageHeight;
        float scoreThreshold;
        std::vector<std::vector<int>> boxIndices, scoreIndices, landmarkIndices;
        std::vector<std::vector<float>> priors;
        [[nodiscard]] std::vector<std::vector<float>> generatePriors(
                const std::vector<std::vector<int>>& minSizes,
                const std::vector<int>& steps) const;
        static EulerAngle calculateFaceAngle(const Point& leftEye, const Point& rightEye,
                                             const Point& noseTip, const Point& leftMouth,
                                             const Point& rightMouth);
        static float iou(const Rect& a, const Rect& b);
    };
}
#endif //FACE_DETECTION_POSTPROCESSING_H