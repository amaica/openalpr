package com.openalpr.jni;


import com.openalpr.jni.json.JSONException;
import com.openalpr.jni.json.JSONObject;

public class AlprCoordinate {
    private final double x;
    private final double y;

    AlprCoordinate(JSONObject coordinateObj) throws JSONException
    {
        x = coordinateObj.getDouble("x");
        y = coordinateObj.getDouble("y");
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public Point2D toPoint() {
        return new Point2D(x, y);
    }
}
