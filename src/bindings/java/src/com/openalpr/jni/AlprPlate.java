package com.openalpr.jni;


import com.openalpr.jni.json.JSONException;
import com.openalpr.jni.json.JSONObject;
import com.openalpr.jni.json.JSONArray;

import java.util.ArrayList;
import java.util.List;

public class AlprPlate {
    private final String characters;
    private final float overall_confidence;
    private final boolean matches_template;
    private final List<CharacterResult> charactersDetailed;

    AlprPlate(JSONObject plateObj, int imgWidth, int imgHeight) throws JSONException
    {
        characters = plateObj.getString("plate");
        overall_confidence = (float) plateObj.getDouble("confidence");
        matches_template = plateObj.getInt("matches_template") != 0;
        JSONArray details = plateObj.optJSONArray("char_details");
        if (details != null) {
            charactersDetailed = new ArrayList<CharacterResult>(details.length());
            for (int i = 0; i < details.length(); i++) {
                charactersDetailed.add(new CharacterResult((JSONObject) details.get(i), imgWidth, imgHeight));
            }
        } else {
            charactersDetailed = new ArrayList<CharacterResult>(0);
        }
    }

    public String getCharacters() {
        return characters;
    }

    public float getOverallConfidence() {
        return overall_confidence;
    }

    public boolean isMatchesTemplate() {
        return matches_template;
    }

    public List<CharacterResult> getCharactersDetailed() {
        return charactersDetailed;
    }
}
