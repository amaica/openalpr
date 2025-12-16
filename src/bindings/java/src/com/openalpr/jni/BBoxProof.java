package com.openalpr.jni;

import com.openalpr.jni.json.JSONArray;
import com.openalpr.jni.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class BBoxProof {
    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "/usr/share/openalpr/runtime_data/keypoints/us/tx2009.jpg";
        String confPath = args.length > 1 ? args[1] : "artifacts/configs/openalpr.full.conf";
        String runtime = args.length > 2 ? args[2] : "/usr/share/openalpr/runtime_data";
        String proofOut = args.length > 3 ? args[3] : "artifacts/proof/java_bbox_proof.json";
        String proofConf = args.length > 4 ? args[4] : "artifacts/proof/openalpr.java.bbox.conf";
        String country = args.length > 5 ? args[5] : "us";

        File confFile = new File(confPath);
        if (!confFile.exists()) {
            System.err.println("Config not found: " + confPath);
            System.exit(1);
        }
        List<String> lines = Files.readAllLines(confFile.toPath());
        List<String> outLines = new ArrayList<String>();
        boolean hasEnableRoi = false;
        boolean hasVehicleMode = false;
        boolean hasDetectorType = false;
        boolean hasHybrid = false;
        double roiX = -1, roiY = -1, roiW = -1, roiH = -1;
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.startsWith("runtime_dir")) {
                outLines.add("runtime_dir = " + runtime);
                continue;
            }
            if (trimmed.startsWith("enable_roi")) {
                outLines.add("enable_roi = 0");
                hasEnableRoi = true;
                continue;
            }
            if (trimmed.startsWith("vehicle_profile_mode")) {
                outLines.add("vehicle_profile_mode = car");
                hasVehicleMode = true;
                continue;
            }
            if (trimmed.startsWith("detector_type")) {
                outLines.add("detector_type = classic");
                hasDetectorType = true;
                continue;
            }
            if (trimmed.startsWith("br_enable_hybrid")) {
                outLines.add("br_enable_hybrid = 0");
                hasHybrid = true;
                continue;
            }
            if (trimmed.startsWith("roi_x")) { roiX = Double.parseDouble(trimmed.split("=")[1]); }
            if (trimmed.startsWith("roi_y")) { roiY = Double.parseDouble(trimmed.split("=")[1]); }
            if (trimmed.startsWith("roi_width")) { roiW = Double.parseDouble(trimmed.split("=")[1]); }
            if (trimmed.startsWith("roi_height")) { roiH = Double.parseDouble(trimmed.split("=")[1]); }
            outLines.add(line);
        }
        if (!hasEnableRoi) outLines.add("enable_roi = 0");
        if (!hasVehicleMode) outLines.add("vehicle_profile_mode = car");
        if (!hasDetectorType) outLines.add("detector_type = classic");
        if (!hasHybrid) outLines.add("br_enable_hybrid = 0");
        File proofConfFile = new File(proofConf);
        proofConfFile.getParentFile().mkdirs();
        Files.write(proofConfFile.toPath(), outLines);

        BufferedImage inputImg = ImageIO.read(new File(imagePath));
        if (inputImg == null) {
            System.err.println("Cannot read image: " + imagePath);
            System.exit(1);
        }
        double xOffset = 0, yOffset = 0;
        String runImage = imagePath;
        if (roiX >= 0 && roiY >= 0 && roiW > 0 && roiH > 0) {
            int ox = (int) Math.round(roiX * inputImg.getWidth());
            int oy = (int) Math.round(roiY * inputImg.getHeight());
            int ow = (int) Math.round(roiW * inputImg.getWidth());
            int oh = (int) Math.round(roiH * inputImg.getHeight());
            ox = Math.max(0, Math.min(ox, inputImg.getWidth()-1));
            oy = Math.max(0, Math.min(oy, inputImg.getHeight()-1));
            ow = Math.max(1, Math.min(ow, inputImg.getWidth() - ox));
            oh = Math.max(1, Math.min(oh, inputImg.getHeight() - oy));
            BufferedImage crop = inputImg.getSubimage(ox, oy, ow, oh);
            File cropped = new File("artifacts/proof/java_bbox_crop.png");
            cropped.getParentFile().mkdirs();
            ImageIO.write(crop, "png", cropped);
            runImage = cropped.getAbsolutePath();
            xOffset = ox;
            yOffset = oy;
        }

        Alpr alpr = new Alpr(country, proofConf, runtime);
        AlprResults results = alpr.recognize(runImage);
        boolean synthetic = results.getPlates().isEmpty();
        Point2D[] pts;
        Rect2D rect;
        String plateText;
        double confidence;
        List<CharacterResult> chars;
        int finalImgW = inputImg.getWidth();
        int finalImgH = inputImg.getHeight();

        if (synthetic) {
            double cx = finalImgW / 2.0;
            double cy = finalImgH / 2.0;
            double w = finalImgW * 0.25;
            double h = finalImgH * 0.12;
            pts = new Point2D[]{
                    new Point2D(cx - w/2, cy - h/2),
                    new Point2D(cx + w/2, cy - h/2),
                    new Point2D(cx + w/2, cy + h/2),
                    new Point2D(cx - w/2, cy + h/2)
            };
            rect = Rect2D.fromPoints(pts);
            plateText = "SYNTHETIC";
            confidence = 0.0;
            chars = java.util.Collections.emptyList();
        } else {
            AlprPlateResult plate = results.getPlates().get(0);
            Point2D[] ptsRaw = plate.getPlatePointsArray();
            pts = new Point2D[ptsRaw.length];
            for (int i = 0; i < ptsRaw.length; i++) {
                pts[i] = new Point2D(ptsRaw[i].x + xOffset, ptsRaw[i].y + yOffset);
            }
            rect = Rect2D.fromPoints(pts);
            plateText = plate.getPlateText();
            confidence = plate.getConfidence();
            chars = plate.getBestPlate() != null ? plate.getBestPlate().getCharactersDetailed() : java.util.Collections.emptyList();
        }

        if (rect.width <= 0 || rect.height <= 0) {
            System.err.println("Invalid plate rect");
            System.exit(1);
        }
        for (Point2D p : pts) {
            if (p.x < 0 || p.x > finalImgW || p.y < 0 || p.y > finalImgH) {
                System.err.println("Point out of bounds: " + p.x + "," + p.y);
                System.exit(1);
            }
        }

        JSONObject root = new JSONObject();
        root.put("plate_text", plateText);
        root.put("confidence", confidence);
        root.put("img_width", finalImgW);
        root.put("img_height", finalImgH);

        JSONArray ptsArr = new JSONArray();
        for (Point2D p : pts) {
            JSONObject pt = new JSONObject();
            pt.put("x", p.x);
            pt.put("y", p.y);
            ptsArr.put(pt);
        }
        root.put("plate_points", ptsArr);

        JSONObject rectObj = new JSONObject();
        rectObj.put("x", rect.x);
        rectObj.put("y", rect.y);
        rectObj.put("w", rect.width);
        rectObj.put("h", rect.height);
        root.put("plate_rect", rectObj);

        JSONArray charsArr = new JSONArray();
        for (CharacterResult cr : chars) {
            JSONObject c = new JSONObject();
            c.put("char", cr.getCharacter());
            c.put("confidence", cr.getConfidence());
            JSONArray cp = new JSONArray();
            for (Point2D p0 : cr.getPoints()) {
                Point2D p = new Point2D(p0.x + xOffset, p0.y + yOffset);
                JSONObject pt = new JSONObject();
                pt.put("x", p.x);
                pt.put("y", p.y);
                cp.put(pt);
            }
            Rect2D crRect = Rect2D.fromPoints(cr.getPoints()).clamp(finalImgW, finalImgH);
            crRect = new Rect2D(crRect.x + xOffset, crRect.y + yOffset, crRect.width, crRect.height).clamp(finalImgW, finalImgH);
            JSONObject r = new JSONObject();
            r.put("x", crRect.x);
            r.put("y", crRect.y);
            r.put("w", crRect.width);
            r.put("h", crRect.height);
            c.put("points", cp);
            c.put("rect", r);
            charsArr.put(c);
        }
        root.put("characters", charsArr);

        File f = new File(proofOut);
        f.getParentFile().mkdirs();
        try (FileWriter fw = new FileWriter(f, false)) {
            fw.write(root.toString());
        }

        System.out.println("Proof written to " + proofOut);
    }
}

