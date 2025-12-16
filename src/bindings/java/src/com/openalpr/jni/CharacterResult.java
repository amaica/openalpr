package com.openalpr.jni;

import com.openalpr.jni.json.JSONArray;
import com.openalpr.jni.json.JSONException;
import com.openalpr.jni.json.JSONObject;

public class CharacterResult {
    private final String character;
    private final double confidence;
    private final Point2D[] points;
    private final Rect2D rect;

    CharacterResult(JSONObject obj, int imgWidth, int imgHeight) throws JSONException {
        character = obj.getString("char");
        confidence = obj.getDouble("confidence");
        JSONArray ptsArr = obj.optJSONArray("points");
        points = new Point2D[ptsArr != null ? ptsArr.length() : 0];
        for (int i = 0; i < points.length; i++) {
            JSONObject pt = (JSONObject) ptsArr.get(i);
            double x = pt.getDouble("x");
            double y = pt.getDouble("y");
            points[i] = new Point2D(x, y).clamp(imgWidth, imgHeight);
        }
        rect = Rect2D.fromPoints(points).clamp(imgWidth, imgHeight);
    }

    public String getCharacter() {
        return character;
    }

    public double getConfidence() {
        return confidence;
    }

    public Point2D[] getPoints() {
        return points;
    }

    public Rect2D getRect() {
        return rect;
    }
}

