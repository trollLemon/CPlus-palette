#include "kmean.h"
#include "dataTypes.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
double colorDistance(Point &p, Point &q) {
    std::vector<int> pRgb = p.getRGB();
    std::vector<int> qRgb = q.getRGB();

    int r = pRgb[0] - qRgb[0];
    int g = pRgb[1] - qRgb[1];
    int b = pRgb[2] - qRgb[2];

    return sqrt(r * r + g * g + b * b);
}

void chooseCentroids(std::vector<Cluster> &clusters, std::vector<Point> &points,
                     int k) {

    std::vector<int> indecies;
    for (uint i = 0; i < points.size(); ++i) {
        indecies.push_back(i);
    }
    std::random_device rd;
    std::seed_seq ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    std::mt19937 gen{ss};
    std::ranges::shuffle(indecies, gen);
    for (int i{0}; i < k; ++i) {
        int randomIndex = indecies.at(i);
        Cluster clstr{Cluster(points.at(randomIndex), i)};
        clusters.push_back(clstr);
    }
}

void assignPoints(std::vector<Point> &points, std::vector<Cluster> &clusters) {

    for (Point &p : points) {

        int closestId{clusters.at(0).getId()};

        double distance{colorDistance(p, clusters.at(0).getCentroid())};

        for (Cluster &c : clusters) {
            double currDist{colorDistance(p, c.getCentroid())};

            if (currDist < distance) {
                currDist = distance;
                closestId = c.getId();
            }
        }

        for (int i{0}; i < clusters.size(); ++i) {
            if (clusters[i].getId() == closestId) {
                clusters[i].addPoint(p);
                break;
            }
        }
    }
}

void updateCentroids(std::vector<Cluster> &clusters) {
    for (Cluster &c : clusters) {
        c.calculateNewCentroid();
    }
}

bool done(std::array<int, 3> &a, std::array<int, 3> &b) {

    if (abs(a[0] - b[0]) == 0 && abs(a[1] - b[1]) == 0 && abs(a[2] - b[2]) == 0)
        return true;
    else
        return false;
}

std::vector<Cluster> generatePalette(std::vector<std::array<int, 3>> &colorData,
                                   int size) {
    // load image data into points, then put them in the points vector
    std::vector<Point> points;
    std::vector<Cluster> clusters;

    for (std::array<int, 3> &rgb : colorData) {
        Point p{Point(rgb[0], rgb[1], rgb[2])};

        points.push_back(p);
    }

    // create clusters and choose some starting centroids
    chooseCentroids(clusters, points, size);
    while (true) {
    
        std::array<int, 3> oldRgb{};

        for (Cluster &c : clusters) {

            std::vector<int> rgb = c.getCentroid().getRGB();
            oldRgb[0] += rgb[0];
            oldRgb[1] += rgb[1];
            oldRgb[2] += rgb[2];
        }

        assignPoints(points, clusters);
        updateCentroids(clusters);

        std::array<int, 3> newRgb{};

        for (Cluster &c : clusters) {
            std::vector<int> rgb = c.getCentroid().getRGB();
            newRgb[0] += rgb[0];
            newRgb[1] += rgb[1];
            newRgb[2] += rgb[2];
        }

        if (done(oldRgb, newRgb))
            break;
        else
            continue;
    }


    return clusters;
}
