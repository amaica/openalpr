package com.openalpr.jni;

public class Rect2D {
    public final double x;
    public final double y;
    public final double width;
    public final double height;

    public Rect2D(double x, double y, double width, double height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public static Rect2D fromPoints(Point2D[] pts) {
        if (pts == null || pts.length == 0) return new Rect2D(0, 0, 0, 0);
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
        for (Point2D p : pts) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }
        return new Rect2D(minX, minY, Math.max(0, maxX - minX), Math.max(0, maxY - minY));
    }

    public Rect2D clamp(double maxWidth, double maxHeight) {
        double cx = Math.max(0, Math.min(x, maxWidth));
        double cy = Math.max(0, Math.min(y, maxHeight));
        double cw = Math.max(0, Math.min(width, maxWidth - cx));
        double ch = Math.max(0, Math.min(height, maxHeight - cy));
        return new Rect2D(cx, cy, cw, ch);
    }
}

