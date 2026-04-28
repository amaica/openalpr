package com.openalpr.jni;


import com.openalpr.jni.json.JSONArray;
import com.openalpr.jni.json.JSONException;
import com.openalpr.jni.json.JSONObject;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class AlprPlateResult {
    // The number requested is always >= the topNPlates count
    private final int requested_topn;

    // the best plate is the topNPlate with the highest confidence
    private final AlprPlate bestPlate;

    // A list of possible plate number permutations
    private List<AlprPlate> topNPlates;

    // The processing time for this plate
    private final float processing_time_ms;

    // the X/Y coordinates of the corners of the plate (clock-wise from top-left)
    private List<Point2D> plate_points;
    private Rect2D plate_rect;

    // The index of the plate if there were multiple plates returned
    private final int plate_index;

    // When region detection is enabled, this returns the region.  Region detection is experimental
    private final int regionConfidence;
    private final String region;

    AlprPlateResult(JSONObject plateResult, int imgWidth, int imgHeight) throws JSONException
    {
        requested_topn = plateResult.getInt("requested_topn");

        JSONArray candidatesArray = plateResult.getJSONArray("candidates");

        if (candidatesArray.length() > 0)
            bestPlate = new AlprPlate((JSONObject) candidatesArray.get(0), imgWidth, imgHeight);
        else
            bestPlate = null;

        topNPlates = new ArrayList<AlprPlate>(candidatesArray.length());
        for (int i = 0; i < candidatesArray.length(); i++)
        {
            JSONObject candidateObj = (JSONObject) candidatesArray.get(i);
            AlprPlate newPlate = new AlprPlate(candidateObj, imgWidth, imgHeight);
            topNPlates.add(newPlate);
        }

        JSONArray coordinatesArray = plateResult.getJSONArray("coordinates");
        plate_points = new ArrayList<Point2D>(coordinatesArray.length());
        for (int i = 0; i < coordinatesArray.length(); i++)
        {
            JSONObject coordinateObj = (JSONObject) coordinatesArray.get(i);
            AlprCoordinate coordinate = new AlprCoordinate(coordinateObj);
            plate_points.add(coordinate.toPoint().clamp(imgWidth, imgHeight));
        }
        plate_rect = Rect2D.fromPoints(getPlatePointsArray()).clamp(imgWidth, imgHeight);

        processing_time_ms = (float) plateResult.getDouble("processing_time_ms");
        plate_index = plateResult.getInt("plate_index");

        regionConfidence = plateResult.getInt("region_confidence");
        region = plateResult.getString("region");

    }

    public int getRequestedTopn() {
        return requested_topn;
    }

    public AlprPlate getBestPlate() {
        return bestPlate;
    }

    public List<AlprPlate> getTopNPlates() {
        return topNPlates;
    }

    public String getPlateText() {
        return bestPlate != null ? bestPlate.getCharacters() : "";
    }

    public double getConfidence() {
        return bestPlate != null ? bestPlate.getOverallConfidence() : 0.0;
    }

    public float getProcessingTimeMs() {
        return processing_time_ms;
    }

    public List<Point2D> getPlatePoints() {
        return plate_points;
    }

    public Point2D[] getPlatePointsArray() {
        return plate_points.toArray(new Point2D[plate_points.size()]);
    }

    public Rect2D getPlateRect() {
        return plate_rect;
    }

    public int getPlateIndex() {
        return plate_index;
    }

    public int getRegionConfidence() {
        return regionConfidence;
    }

    public String getRegion() {
        return region;
    }

    public static Point2D[] orderedClockwise(Point2D[] pts) {
        if (pts == null || pts.length == 0) return new Point2D[0];
        double cx = 0, cy = 0;
        for (Point2D p : pts) { cx += p.x; cy += p.y; }
        cx /= pts.length; cy /= pts.length;
        final double centerX = cx;
        final double centerY = cy;
        Point2D[] ordered = pts.clone();
        java.util.Arrays.sort(ordered, (a,b) -> {
            double angA = Math.atan2(a.y - centerY, a.x - centerX);
            double angB = Math.atan2(b.y - centerY, b.x - centerX);
            return Double.compare(angA, angB);
        });
        return ordered;
    }
}
