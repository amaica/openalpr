package com.openalpr.jni;

public class Point2D {
    public final double x;
    public final double y;

    public Point2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Point2D clamp(double maxWidth, double maxHeight) {
        double cx = Math.max(0, Math.min(x, maxWidth));
        double cy = Math.max(0, Math.min(y, maxHeight));
        return new Point2D(cx, cy);
    }
}

